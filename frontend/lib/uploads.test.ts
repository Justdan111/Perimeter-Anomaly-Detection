// Run with: npm test
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { isWholeFrameZone } from "./alerts.ts";
import { type JobStatus, checkFileBeforeUpload, describeJob } from "./uploads.ts";

const MB = 1024 * 1024;

describe("checkFileBeforeUpload", () => {
  // A quick check for a fast answer; the server re-checks everything.
  test("a normal video file passes", () => {
    assert.equal(checkFileBeforeUpload({ name: "clip.mp4", size: 20 * MB, type: "video/mp4" }, 100 * MB), null);
  });

  test("an oversized file is refused before uploading, with both sizes", () => {
    const msg = checkFileBeforeUpload({ name: "big.mp4", size: 150 * MB, type: "video/mp4" }, 100 * MB);
    assert.match(msg ?? "", /150 MB.*100 MB/);
  });

  test("exactly at the limit passes", () => {
    assert.equal(checkFileBeforeUpload({ name: "c.mp4", size: 100 * MB, type: "video/mp4" }, 100 * MB), null);
  });

  test("an obviously non-video file is refused", () => {
    assert.ok(checkFileBeforeUpload({ name: "photo.jpg", size: MB, type: "image/jpeg" }, 100 * MB));
    assert.ok(checkFileBeforeUpload({ name: "notes.txt", size: 10, type: "text/plain" }, 100 * MB));
  });

  test("a video whose browser type is blank is judged by extension", () => {
    // Some browsers report "" for .mov/.mkv; don't refuse a real video.
    assert.equal(checkFileBeforeUpload({ name: "IMG_0001.MOV", size: MB, type: "" }, 100 * MB), null);
    assert.ok(checkFileBeforeUpload({ name: "archive.zip", size: MB, type: "" }, 100 * MB));
  });

  test("an empty file is refused", () => {
    assert.ok(checkFileBeforeUpload({ name: "c.mp4", size: 0, type: "video/mp4" }, 100 * MB));
  });
});

function job(overrides: Partial<JobStatus>): JobStatus {
  return {
    status: "processing",
    frames_to_process: 120,
    frames_processed: 30,
    queue_position: 0,
    error: null,
    ...overrides,
  };
}

describe("describeJob", () => {
  test("processing shows frames done and the fraction", () => {
    const d = describeJob(job({}));
    assert.equal(d.fraction, 0.25);
    assert.match(d.label, /30 of 120 frames/);
  });

  test("queued shows the queue position and no progress", () => {
    const d = describeJob(job({ status: "queued", frames_processed: 0, queue_position: 2 }));
    assert.equal(d.fraction, 0);
    assert.match(d.label, /queue.*2/i);
  });

  test("complete is 100%", () => {
    assert.equal(describeJob(job({ status: "complete", frames_processed: 120 })).fraction, 1);
  });

  test("failed shows the server's reason", () => {
    const d = describeJob(job({ status: "failed", error: "the video could not be read: truncated" }));
    assert.match(d.label, /truncated/);
  });

  test("zero frames to process doesn't divide by zero", () => {
    assert.equal(describeJob(job({ frames_to_process: 0, frames_processed: 0 })).fraction, 0);
  });

  test("fraction never exceeds 1", () => {
    assert.equal(describeJob(job({ frames_to_process: 10, frames_processed: 11 })).fraction, 1);
  });
});

describe("isWholeFrameZone", () => {
  test("the upload default zone is recognised", () => {
    assert.equal(isWholeFrameZone([[0, 0], [1280, 0], [1280, 720], [0, 720]], 1280, 720), true);
  });

  test("the committed sample zone is not", () => {
    const sample: [number, number][] = [[0, 428], [340, 396], [372, 468], [252, 556], [0, 628]];
    assert.equal(isWholeFrameZone(sample, 1280, 720), false);
  });

  test("a rectangle short of the frame edge is not", () => {
    assert.equal(isWholeFrameZone([[0, 0], [1280, 0], [1280, 700], [0, 700]], 1280, 720), false);
  });
});
