"""
Downloads the real Sen1Floods11 hand-labeled India subset (68 Sentinel-1 scenes
+ ground-truth water masks) from the public gs://sen1floods11 bucket over plain
HTTPS (no gcloud/gsutil/auth required) into data/raw/sen1floods11_india/.
"""

import json
import os
import urllib.request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'sen1floods11_india')
BUCKET_BASE = "https://storage.googleapis.com/sen1floods11/"
PREFIXES = {
    'S1Hand': "v1.1/data/flood_events/HandLabeled/S1Hand/India",
    'LabelHand': "v1.1/data/flood_events/HandLabeled/LabelHand/India",
}


def list_objects(prefix):
    url = f"https://storage.googleapis.com/storage/v1/b/sen1floods11/o?prefix={prefix}&maxResults=1000"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return {item['name']: int(item['size']) for item in data.get('items', [])}


def download_folder(folder):
    out_dir = os.path.join(OUT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    remote = list_objects(PREFIXES[folder])
    print(f"[*] {folder}: {len(remote)} files")

    for remote_path, size in remote.items():
        fname = remote_path.split('/')[-1]
        local_path = os.path.join(out_dir, fname)
        if os.path.exists(local_path) and os.path.getsize(local_path) == size:
            continue
        urllib.request.urlretrieve(BUCKET_BASE + remote_path, local_path)

    # Verify sizes, re-download anything truncated by a network interruption.
    for remote_path, size in remote.items():
        fname = remote_path.split('/')[-1]
        local_path = os.path.join(out_dir, fname)
        if os.path.getsize(local_path) != size:
            print(f"    re-downloading truncated file: {fname}")
            urllib.request.urlretrieve(BUCKET_BASE + remote_path, local_path)


if __name__ == '__main__':
    for folder in PREFIXES:
        download_folder(folder)
    print(f"[+] Done. Saved to {OUT_DIR}")
