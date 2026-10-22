"""Download public pretrained inference models, verifying SHA256 before use."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile


ARCFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
CHECKSUMS = {
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
}


def verified(path, expected):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def download(url, path):
    print(f"Downloading {url}", flush=True)
    with urlopen(url, timeout=60) as source, path.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, checksum in CHECKSUMS.items():
        target = args.output / name
        if verified(target, checksum):
            print(f"Verified {target}")
            continue
        with tempfile.TemporaryDirectory(dir=args.output) as temporary:
            candidate = Path(temporary) / name
            if name == "w600k_r50.onnx":
                archive_path = Path(temporary) / "buffalo_l.zip"
                download(ARCFACE_URL, archive_path)
                with ZipFile(archive_path) as archive:
                    member = next(item for item in archive.namelist() if Path(item).name == name)
                    with archive.open(member) as source, candidate.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            else:
                download(YUNET_URL, candidate)
            if not verified(candidate, checksum):
                raise RuntimeError(f"SHA256 mismatch for {name}; existing weights were not replaced")
            candidate.replace(target)
            print(f"Verified {target}")


if __name__ == "__main__":
    main()
