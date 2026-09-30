"""Local camera agent (Phase 4, Stage 4a — validation, see docs/PHASE4.md).

Runs next to the camera. Captures frames with OpenCV, samples them at a fixed
rate, and POSTs each as a JPEG to the backend's `/live/frames`. It also polls
`/live/stats` so a sustained run leaves a record of the server's memory and
counters over time, and notices a restart (the server's `process_started_at`
changes).

    cd backend
    uv run python ../agent/live_agent.py --url https://<service>.onrender.com \\
        --source 0 --minutes 15 --out ../agent/runs/run1

`--source` is a camera index (0 = the built-in webcam) or a video file, which
is looped in real time as a stand-in camera.

Sampling: 1 frame per second by default. On the free tier, detection alone
costs about 0.55 s per 720p frame (the sample clip: ~67 s for 120 frames),
so 2 fps would need the whole CPU just to keep up — no headroom for uploads
and nothing left if a frame is slow. At 1 fps a person walking in at
~1.4 m/s is still seen several times while crossing a room-sized view.

Design points:
- A capture thread reads continuously and keeps only the latest frame, so a
  sent frame is always current (webcams buffer; reading only when sending
  would send stale frames).
- One request in flight, never a queue: if the server takes longer than the
  interval, the next frame goes as soon as the last one returns.
- Errors (server down, 503 busy) are logged and the loop keeps going, so the
  run records outages instead of ending at the first one.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2


class LatestFrame:
    """Reads a source continuously; `get()` returns the newest frame."""

    def __init__(self, source: str) -> None:
        self.is_file = not source.isdigit()
        self.cap = cv2.VideoCapture(source if self.is_file else int(source))
        if not self.cap.isOpened():
            raise SystemExit(
                f"could not open source {source!r}"
                + ("" if self.is_file else " (on macOS, allow camera access for this terminal app)")
            )
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.file_interval = 1 / fps if self.is_file and fps and fps > 0 else 0
        self._frame = None
        self._lock = threading.Lock()
        self.stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:  # loop the file as a stand-in camera
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
            if self.file_interval:
                time.sleep(self.file_interval)  # real-time pace, like a camera

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self.stopped = True
        self.cap.release()


def post_frame(url: str, jpeg: bytes, camera: str, sent_at: float, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{url}/live/frames?camera={camera}&sent_at={sent_at:.3f}",
        data=jpeg,
        headers={"Content-Type": "image/jpeg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:  # connection refused/reset, timeout, 502 page...
        return 0, {"error": f"{type(e).__name__}: {e}"}


def get_stats(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/live/stats", timeout=20) as r:
            return json.loads(r.read())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="backend base URL")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sent per second (default 1)")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--camera", default="webcam", help="name reported with each frame")
    ap.add_argument("--max-width", type=int, default=1280, help="downscale wider frames")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--stats-every", type=float, default=15.0, help="seconds between /live/stats polls")
    ap.add_argument("--out", default="agent/runs/latest", help="folder for frames.csv and stats.csv")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cam = LatestFrame(a.source)
    while cam.get() is None:
        time.sleep(0.05)

    frames_f = open(out / "frames.csv", "w", newline="")
    frames = csv.writer(frames_f)
    frames.writerow(["t_s", "status", "rtt_ms", "detect_ms", "lock_wait_ms", "alerts", "detail"])
    stats_f = open(out / "stats.csv", "w", newline="")
    stats_w = None

    interval = 1 / a.fps
    start = time.time()
    end = start + a.minutes * 60
    next_send = start
    next_stats = start
    seen_start: float | None = None
    counts: dict[str, int] = {}
    try:
        while time.time() < end:
            now = time.time()
            if now >= next_stats:
                s = get_stats(url)
                row = {"t_s": round(now - start, 1), **(s or {"unreachable": True})}
                if s:
                    if seen_start is not None and s["process_started_at"] != seen_start:
                        print(f"[{row['t_s']:7.1f}s] SERVER RESTARTED (process_started_at changed)")
                        row["restart_detected"] = True
                    seen_start = s["process_started_at"]
                if stats_w is None:
                    stats_w = csv.DictWriter(stats_f, fieldnames=list(row) + ["restart_detected", "unreachable"],
                                             extrasaction="ignore")
                    stats_w.writeheader()
                stats_w.writerow(row)
                stats_f.flush()
                print(f"[{row['t_s']:7.1f}s] stats: " + (
                    f"mem {s.get('cgroup_current_mb')}/{s.get('cgroup_peak_mb')} MB (cur/peak), "
                    f"processed {s['frames_processed']}, dropped busy {s['frames_dropped_model_busy']}, "
                    f"detect ~{s['recent_detect_ms_median']} ms, uploads active {s['upload_jobs_active']}"
                    if s else "UNREACHABLE"), flush=True)
                next_stats = now + a.stats_every

            if now < next_send:
                time.sleep(min(next_send, next_stats) - now)
                continue
            next_send = now + interval
            frame = cam.get()
            h, w = frame.shape[:2]
            if w > a.max_width:
                frame = cv2.resize(frame, (a.max_width, round(h * a.max_width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            t0 = time.time()
            status, body = post_frame(url, buf.tobytes(), a.camera, t0, timeout=15)
            rtt = (time.time() - t0) * 1000
            key = str(status)
            counts[key] = counts.get(key, 0) + 1
            frames.writerow([
                round(t0 - start, 2), status, round(rtt), body.get("detect_ms"), body.get("lock_wait_ms"),
                len(body.get("alerts", [])) if status == 200 else "",
                "" if status == 200 else str(body.get("detail") or body.get("error", ""))[:120],
            ])
            frames_f.flush()
            if status != 200:
                print(f"[{t0 - start:7.1f}s] frame -> {status} {str(body)[:100]}", flush=True)
                if status in (0, 502):
                    time.sleep(2)  # server down: don't hammer it
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cam.stop()
        frames_f.close()
        stats_f.close()
        print(f"done after {(time.time() - start) / 60:.1f} min; responses: {counts}; files in {out}")


if __name__ == "__main__":
    main()
