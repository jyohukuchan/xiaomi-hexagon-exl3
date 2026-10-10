"""Extract the official SDK addon into an ignored build directory, without redistribution."""
import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import zipfile


def validate(archive):
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts or ":" in entry.filename or "\\" in entry.filename:
            raise ValueError("Unsafe archive member")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if digest != "365ea693b279c4f267a098b1ce8c4e717dca763e8c2f156aa4269aaba62d8ea9":
        raise ValueError("Unexpected official archive SHA256; this extractor is pinned to the tested release")
    with zipfile.ZipFile(args.archive) as outer:
        validate(outer)
        with zipfile.ZipFile(io.BytesIO(outer.read("hexkl-1.0.0-beta1-6.4.0.0.zip"))) as inner:
            validate(inner)
            args.destination.mkdir(parents=True, exist_ok=False)
            inner.extractall(args.destination)
    print("archive SHA256", digest)
    print("addon", args.destination.resolve() / "hexkl_addon")


if __name__ == "__main__":
    main()
