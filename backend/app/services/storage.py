"""Where upload results live: Cloudflare R2 in production, a folder locally.

Why R2 (docs/PHASE1.md): Render's free tier has no persistent disk. The
filesystem is wiped on every redeploy, restart and idle spin-down, so
anything written there is gone the next time someone checks on a job. R2 is
S3-compatible, free up to 10 GB with no egress fees, and is used through
boto3 pointed at R2's endpoint.

What goes in the store, per upload job (prefix `jobs/<job_id>/`):
    job.json          status, clip facts, zone, classes
    result.json       the alerts (each alert's `snapshot` is an object key)
    reference.jpg     the upload's first frame, for the zone view
    snapshots/*.jpg   one per frame that produced an alert
The uploaded video itself is never stored here — it stays on local disk
only while its job runs (the doc's "temporary/local" allowance).

Snapshots and the reference frame are served as **signed, expiring links**
straight from R2 (the bucket stays private): 6 hours by default, long enough
to review a large job; reloading the page issues fresh links.

The local store exists so development and tests don't need a bucket. It is
explicitly not durable, and /health reports which store is in use.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_URL_EXPIRY_S = 6 * 60 * 60

R2_ENV = (
    "PERIMETER_R2_ENDPOINT",
    "PERIMETER_R2_BUCKET",
    "PERIMETER_R2_ACCESS_KEY_ID",
    "PERIMETER_R2_SECRET_ACCESS_KEY",
)


class StorageError(RuntimeError):
    """The result store is misconfigured or unreachable."""


class ObjectStore(Protocol):
    kind: str

    def put_file(self, key: str, path: Path, content_type: str) -> None: ...
    def put_json(self, key: str, data: dict) -> None: ...
    def get_json(self, key: str) -> dict | None: ...
    def exists(self, key: str) -> bool: ...
    def url(self, key: str) -> str: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def check(self) -> None: ...


class S3ObjectStore:
    """Any S3-compatible bucket; Cloudflare R2 in production."""

    kind = "r2"

    def __init__(self, client, bucket: str, url_expiry_s: int = DEFAULT_URL_EXPIRY_S) -> None:
        self._client = client
        self.bucket = bucket
        self.url_expiry_s = url_expiry_s

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        self._client.upload_file(
            str(path), self.bucket, key, ExtraArgs={"ContentType": content_type}
        )

    def put_json(self, key: str, data: dict) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data).encode(),
            ContentType="application/json",
        )

    def get_json(self, key: str) -> dict | None:
        try:
            body = self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(body)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def url(self, key: str) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.url_expiry_s,
        )

    def delete_prefix(self, prefix: str) -> None:
        # list_objects_v2 returns at most 1000 keys per page; a long clip can
        # produce more snapshots than that.
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

    def check(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except (ClientError, BotoCoreError) as e:
            raise StorageError(f"R2 bucket {self.bucket!r} is not reachable: {e}") from e


class LocalObjectStore:
    """A folder standing in for the bucket. NOT durable on the host."""

    kind = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"key escapes the store: {key!r}")
        return path

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    def put_json(self, key: str, data: dict) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(target)  # atomic: a reader never sees half a file

    def get_json(self, key: str) -> dict | None:
        path = self._path(key)
        return json.loads(path.read_text()) if path.is_file() else None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def url(self, key: str) -> str:
        # Served by the API's /jobs/{id}/... routes, which mirror the keys.
        return f"/{key}"

    def local_path(self, key: str) -> Path:
        return self._path(key)

    def delete_prefix(self, prefix: str) -> None:
        target = self._path(prefix.rstrip("/"))
        shutil.rmtree(target, ignore_errors=True)

    def check(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def store_from_env(local_root: Path) -> ObjectStore:
    """R2 if configured, else a local folder.

    All four R2 variables or none: a partial set is an error, because falling
    back to local storage on the host would silently lose every result at
    the next restart.
    """
    present = {name: os.environ.get(name) for name in R2_ENV}
    if not any(present.values()):
        return LocalObjectStore(local_root)
    missing = [name for name, value in present.items() if not value]
    if missing:
        raise StorageError(f"R2 is partly configured; missing: {', '.join(missing)}")
    client = boto3.client(
        "s3",
        endpoint_url=present["PERIMETER_R2_ENDPOINT"],
        aws_access_key_id=present["PERIMETER_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=present["PERIMETER_R2_SECRET_ACCESS_KEY"],
        region_name="auto",  # R2 ignores regions; "auto" is its documented value
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )
    return S3ObjectStore(
        client=client,
        bucket=present["PERIMETER_R2_BUCKET"],
        url_expiry_s=int(os.environ.get("PERIMETER_SIGNED_URL_EXPIRY_S", DEFAULT_URL_EXPIRY_S)),
    )
