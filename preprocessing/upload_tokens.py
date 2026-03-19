import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import TOKEN_DIR, AG_TOKEN_DIR, S3_BUCKET, S3_PREFIX

SPLITS = ("train", "val", "test")
CODEC_DIRS = {
    "snac": TOKEN_DIR,
    "encodec": Path(AG_TOKEN_DIR),
}


def _s3_key(prefix: str, codec: str, split: str, filename: str) -> str:
    parts = [p for p in (prefix, codec, split, filename) if p]
    return "/".join(parts)


def upload_tokens(
    codec: str,
    splits: list[str],
    bucket: str = S3_BUCKET,
    prefix: str = S3_PREFIX,
    token_dir: Path | None = None,
):
    base_dir = token_dir or CODEC_DIRS[codec]
    base_dir = Path(base_dir)
    s3 = boto3.client("s3")

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "404":
            raise SystemExit(f"Bucket '{bucket}' does not exist.") from exc
        if code == "403":
            raise SystemExit(f"No access to bucket '{bucket}'.") from exc
        raise

    for split in splits:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            print(f"[skip] {split_dir} does not exist")
            continue

        for path in sorted(split_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(base_dir)
            key = _s3_key(prefix, codec, str(rel.parent), rel.name)
            print(f"  {path} -> s3://{bucket}/{key}")
            s3.upload_file(str(path), bucket, key)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload token files to S3")
    parser.add_argument(
        "--codec",
        choices=list(CODEC_DIRS),
        required=True,
        help="Which codec's tokens to upload",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        nargs="+",
        default=None,
        help="Splits to upload (default: all)",
    )
    parser.add_argument("--bucket", default=S3_BUCKET)
    parser.add_argument("--prefix", default=S3_PREFIX)
    parser.add_argument(
        "--token-dir",
        default=None,
        help="Override the local token directory",
    )
    args = parser.parse_args()

    upload_tokens(
        codec=args.codec,
        splits=args.split or list(SPLITS),
        bucket=args.bucket,
        prefix=args.prefix,
        token_dir=Path(args.token_dir) if args.token_dir else None,
    )
