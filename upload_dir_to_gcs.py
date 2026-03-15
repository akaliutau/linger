#!/usr/bin/env python3
from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

from google.cloud import storage


def split_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--project")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.exists():
        raise SystemExit(f"Local directory does not exist: {local_dir}")

    bucket_name, prefix = split_gs_uri(args.output_uri)
    client = storage.Client(project=args.project or None)
    bucket = client.bucket(bucket_name)

    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        blob = bucket.blob(blob_name)
        content_type, _ = mimetypes.guess_type(str(path))
        blob.upload_from_filename(str(path), content_type=content_type or "application/octet-stream")
        print(f"uploaded gs://{bucket_name}/{blob_name}")


if __name__ == "__main__":
    main()
