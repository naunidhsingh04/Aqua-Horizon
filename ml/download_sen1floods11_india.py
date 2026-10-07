"""
Downloads the real Sen1Floods11 hand-labeled subset for India (68 Sentinel-1 scenes
+ ground-truth water masks; Pakistan, Sri-Lanka and USA can be named on the command line) from the public gs://sen1floods11 bucket over plain
HTTPS (no gcloud/gsutil/auth required) into data/raw/sen1floods11_india/.
"""

import json
import os
import sys
import urllib.request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUCKET_BASE = "https://storage.googleapis.com/sen1floods11/"

# Country (flood event) name in the bucket -> local folder. India is the default; the other
# countries are used by ml/test_other_countries.py to test the India-trained model abroad.
# Usage: python ml/download_sen1floods11_india.py [Pakistan Sri-Lanka USA ...]
COUNTRIES = ['India', 'Pakistan', 'Sri-Lanka', 'USA']
OUT_DIR = None
PREFIXES = {}


def set_country(country):
    global OUT_DIR, PREFIXES
    OUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'sen1floods11_' + country.lower().replace('-', '_'))
    PREFIXES = {
        'S1Hand': f"v1.1/data/flood_events/HandLabeled/S1Hand/{country}",
        'LabelHand': f"v1.1/data/flood_events/HandLabeled/LabelHand/{country}",
    }


# Asks Google Cloud Storage for the files under a folder; returns {file path: size in bytes}
def list_objects(prefix):
    url = f"https://storage.googleapis.com/storage/v1/b/sen1floods11/o?prefix={prefix}&maxResults=1000"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return {item['name']: int(item['size']) for item in data.get('items', [])}


# Downloads every file of one folder, skipping files already present with the right size
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

    # (size check below compares local file size with the size reported by the bucket)
    # Verify sizes, re-download anything truncated by a network interruption.
    for remote_path, size in remote.items():
        fname = remote_path.split('/')[-1]
        local_path = os.path.join(out_dir, fname)
        if os.path.getsize(local_path) != size:
            print(f"    re-downloading truncated file: {fname}")
            urllib.request.urlretrieve(BUCKET_BASE + remote_path, local_path)


if __name__ == '__main__':
    for country in (sys.argv[1:] or ['India']):
        assert country in COUNTRIES, f"{country} is not one of {COUNTRIES}"
        set_country(country)
        for folder in PREFIXES:
            download_folder(folder)
        print(f"[+] Done. Saved to {OUT_DIR}")
