// Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";

import { API_URL, apiUrl } from "./api.ts";

test("an API path is prefixed with the API's base URL", () => {
  assert.equal(apiUrl("/jobs/abc"), `${API_URL}/jobs/abc`);
});

test("an absolute signed R2 link is used exactly as given", () => {
  // Snapshot URLs from R2 are signed; prefixing (or re-encoding) them would
  // break the signature.
  const signed =
    "https://acct.r2.cloudflarestorage.com/b/jobs/x/snapshots/frame_00001.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc%2Fdef";
  assert.equal(apiUrl(signed), signed);
});
