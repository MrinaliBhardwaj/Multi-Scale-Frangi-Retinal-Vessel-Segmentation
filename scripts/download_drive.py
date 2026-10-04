"""Download the DRIVE dataset (40 images + expert masks + FOV masks) into data/DRIVE.

The official source (https://drive.grand-challenge.org) requires registration. This script pulls an
unmodified copy of the original 2004 release from a public GitHub mirror. If the mirror is
unavailable, download DRIVE manually and unzip it so that data/DRIVE/{training,test}/{images,
1st_manual,mask} exist.
"""
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://github.com/hamdan92/retinal-vessel-segmentation/raw/main/DRIVE.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data")


def main():
    target = os.path.join(DEST, "DRIVE")
    if os.path.isdir(os.path.join(target, "test", "1st_manual")):
        print("DRIVE already present at", target)
        return
    print("Downloading", URL)
    try:
        data = urllib.request.urlopen(URL, timeout=120).read()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed ({exc}). Download DRIVE manually into {target}.")
    os.makedirs(DEST, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(DEST)
    n = sum(len(f) for _, _, f in os.walk(target))
    print(f"Extracted {n} files to {target}")


if __name__ == "__main__":
    main()
