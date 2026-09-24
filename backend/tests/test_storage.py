"""Result storage: Cloudflare R2 in production, a local folder in development.

The same contract runs against both. The R2 store is exercised against
`moto` — an in-memory fake of the S3 API that R2 implements — so these need
no network and no real bucket.
"""

import json
from urllib.parse import parse_qs, urlparse

import boto3
import pytest
from botocore.config import Config
from moto import mock_aws

from app.services.storage import (
    LocalObjectStore,
    S3ObjectStore,
    StorageError,
    store_from_env,
)

BUCKET = "perimeter-test"


@pytest.fixture
def s3_store():
    with mock_aws():
        # SigV4, as in production: R2 rejects the older SigV2 signed links
        # that boto3 would otherwise generate for this client.
        client = boto3.client(
            "s3", region_name="us-east-1", config=Config(signature_version="s3v4")
        )
        client.create_bucket(Bucket=BUCKET)
        yield S3ObjectStore(client=client, bucket=BUCKET, url_expiry_s=21600)


@pytest.fixture
def local_store(tmp_path):
    return LocalObjectStore(tmp_path / "store")


@pytest.fixture(params=["local", "s3"])
def store(request):
    return request.getfixturevalue(f"{request.param}_store")


class TestContract:
    def test_json_round_trip(self, store):
        store.put_json("jobs/abc/job.json", {"status": "complete", "n": 3})
        assert store.get_json("jobs/abc/job.json") == {"status": "complete", "n": 3}

    def test_missing_json_is_none_not_an_error(self, store):
        assert store.get_json("jobs/nope/job.json") is None

    def test_overwrite_replaces(self, store):
        store.put_json("k.json", {"v": 1})
        store.put_json("k.json", {"v": 2})
        assert store.get_json("k.json") == {"v": 2}

    def test_file_upload_and_existence(self, store, tmp_path):
        f = tmp_path / "frame.jpg"
        f.write_bytes(b"\xff\xd8jpegbytes")
        store.put_file("jobs/abc/snapshots/frame_00001.jpg", f, "image/jpeg")
        assert store.exists("jobs/abc/snapshots/frame_00001.jpg")
        assert not store.exists("jobs/abc/snapshots/frame_00002.jpg")

    def test_delete_prefix_removes_only_that_job(self, store, tmp_path):
        f = tmp_path / "x.jpg"
        f.write_bytes(b"x")
        store.put_file("jobs/aaa/snapshots/frame_00001.jpg", f, "image/jpeg")
        store.put_json("jobs/aaa/job.json", {})
        store.put_json("jobs/aab/job.json", {})  # shares a prefix string, different job
        store.delete_prefix("jobs/aaa/")
        assert not store.exists("jobs/aaa/job.json")
        assert not store.exists("jobs/aaa/snapshots/frame_00001.jpg")
        assert store.exists("jobs/aab/job.json")

    def test_check_passes_when_usable(self, store):
        store.check()  # no exception


class TestS3Specifics:
    def test_url_is_a_signed_expiring_link_to_the_private_object(self, s3_store):
        url = s3_store.url("jobs/abc/snapshots/frame_00001.jpg")
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        assert parsed.scheme == "https"
        assert parsed.path.endswith("/jobs/abc/snapshots/frame_00001.jpg")
        assert "X-Amz-Signature" in query
        assert query["X-Amz-Expires"] == ["21600"]  # 6 hours

    def test_json_is_stored_as_json(self, s3_store):
        s3_store.put_json("jobs/abc/job.json", {"a": 1})
        obj = s3_store._client.get_object(Bucket=BUCKET, Key="jobs/abc/job.json")
        assert obj["ContentType"] == "application/json"
        assert json.loads(obj["Body"].read()) == {"a": 1}

    def test_check_fails_clearly_when_the_bucket_is_unreachable(self):
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")  # bucket never created
            store = S3ObjectStore(client=client, bucket="missing-bucket")
            with pytest.raises(StorageError, match="missing-bucket"):
                store.check()

    def test_delete_prefix_handles_more_than_one_listing_page(self, s3_store):
        # S3 lists at most 1000 keys per call; a long clip can have more
        # snapshots than that. Deleting must page through all of them.
        for i in range(1005):
            s3_store._client.put_object(Bucket=BUCKET, Key=f"jobs/big/snapshots/{i:05d}.jpg", Body=b"x")
        s3_store.delete_prefix("jobs/big/")
        left = s3_store._client.list_objects_v2(Bucket=BUCKET, Prefix="jobs/big/")
        assert left.get("KeyCount", 0) == 0


class TestLocalSpecifics:
    def test_url_is_the_api_route_for_the_key(self, local_store):
        assert local_store.url("jobs/abc/snapshots/frame_00001.jpg") == "/jobs/abc/snapshots/frame_00001.jpg"

    def test_keys_cannot_escape_the_store_directory(self, local_store):
        with pytest.raises(ValueError):
            local_store.put_json("../outside.json", {})


class TestStoreFromEnv:
    ENV = {
        "PERIMETER_R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
        "PERIMETER_R2_BUCKET": "b",
        "PERIMETER_R2_ACCESS_KEY_ID": "id",
        "PERIMETER_R2_SECRET_ACCESS_KEY": "secret",
    }

    def test_all_r2_settings_give_an_r2_store(self, monkeypatch, tmp_path):
        for k, v in self.ENV.items():
            monkeypatch.setenv(k, v)
        store = store_from_env(tmp_path)
        assert store.kind == "r2"
        assert store.bucket == "b"

    def test_no_r2_settings_give_a_local_store(self, monkeypatch, tmp_path):
        for k in self.ENV:
            monkeypatch.delenv(k, raising=False)
        assert store_from_env(tmp_path).kind == "local"

    def test_production_client_signs_links_with_sigv4_for_six_hours(self, monkeypatch, tmp_path):
        # R2 only accepts SigV4. Found when a test client without the s3v4
        # setting produced SigV2 links (Signature=/Expires=) — so the
        # production client's setting is pinned here.
        for k, v in self.ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("PERIMETER_SIGNED_URL_EXPIRY_S", raising=False)
        url = store_from_env(tmp_path).url("jobs/abc/reference.jpg")
        query = parse_qs(urlparse(url).query)
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert query["X-Amz-Expires"] == ["21600"]
        assert url.startswith("https://acct.r2.cloudflarestorage.com/b/jobs/abc/reference.jpg?")

    def test_partial_r2_settings_are_an_error_not_a_silent_local_fallback(self, monkeypatch, tmp_path):
        # A typo in one variable on the host must not quietly switch to
        # local storage, where results vanish on the next restart.
        for k, v in self.ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("PERIMETER_R2_SECRET_ACCESS_KEY")
        with pytest.raises(StorageError, match="PERIMETER_R2_SECRET_ACCESS_KEY"):
            store_from_env(tmp_path)
