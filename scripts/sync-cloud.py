"""
CloudFoundry (Nextcloud) Content Sync for CDAC NLP & CV
======================================================
Connects to the course cloud share, checks for new lecture slides,
code, zips, or assignments, and automatically places them into
the structured repository folders without manual dragging and dropping.

Usage:
    python scripts/sync-cloud.py
"""

import os
import re
import sys
import zipfile
import urllib.parse
import xml.etree.ElementTree as ET
import requests

# Base configuration
CLOUD_BASE_URL = "https://cf.cdacb.in/public.php/webdav/002_Modulewise%20Contents/006_Natural%20Language%20Processing%20%26%20Computer%20Vision/"
AUTH = ("l0rOWydcrtrzkiI", "AI@EC2")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_remote_files():
    """List all files in the course folder via WebDAV PROPFIND."""
    headers = {"Depth": "infinity"}
    resp = requests.request("PROPFIND", CLOUD_BASE_URL, auth=AUTH, headers=headers, timeout=25)
    if resp.status_code not in (200, 207):
        print(f"[ERROR] Failed to connect to CloudFoundry: Status {resp.status_code}")
        return []

    root = ET.fromstring(resp.content)
    files = []
    base_prefix = "/public.php/webdav/002_Modulewise Contents/006_Natural Language Processing & Computer Vision/"

    for elem in root.iter("{DAV:}href"):
        unquoted = urllib.parse.unquote(elem.text)
        if unquoted.endswith("/"):
            continue  # skip directory markers
        rel_path = unquoted.replace(base_prefix, "").strip("/")
        if rel_path:
            files.append(rel_path)

    return sorted(list(set(files)))


def determine_destination(filename):
    """
    Decide where a file belongs based on its name and extension:
    - Assignments -> assignments/assignment-0X/
    - DAY 2 code -> notes/day-02/code/DAY 2/
    - Session slides (.pptx) -> notes/day-0X/slides
    - Session guides (.pdf) -> notes/day-0X/pdfs
    - Textbooks -> reference-books/
    - Datasets / zips -> raw-downloads/ & notes/day-0X/
    """
    fname_lower = filename.lower()
    base_name = os.path.basename(filename)

    # 1. Assignments (.docx, .pdf)
    if "assignment" in fname_lower:
        day_match = re.search(r"day[\s_]*(\d+)", fname_lower)
        day_num = int(day_match.group(1)) if day_match else 1
        assign_dir = os.path.join(REPO_ROOT, "assignments", f"assignment-{day_num:02d}")
        os.makedirs(assign_dir, exist_ok=True)
        return os.path.join(assign_dir, base_name)

    # Detect day folder (check day-specific folder prefix first)
    if filename.startswith("DAY 4/") or "day 4" in fname_lower or "day-04" in fname_lower or "session5" in fname_lower or "session6" in fname_lower:
        target_day = "day-04"
    elif filename.startswith("Day 3/") or "day 3" in fname_lower or "day-03" in fname_lower or "session4" in fname_lower:
        target_day = "day-03"
    elif filename.startswith("DAY 2/") or "day 2" in fname_lower or "day-02" in fname_lower or "session3" in fname_lower:
        target_day = "day-02"
    elif filename.startswith("DAY 1/") or "day 1" in fname_lower or "day-01" in fname_lower or "session1" in fname_lower or "session2" in fname_lower:
        target_day = "day-01"
    else:
        target_day = "day-02"

    # 2. Session slides (.pptx)
    if filename.endswith(".pptx"):
        return os.path.join(REPO_ROOT, "notes", target_day, "slides", base_name)

    # 3. Session PDF guides (.pdf)
    if filename.endswith(".pdf"):
        if filename.startswith("DAY 4/") or "probability" in fname_lower:
            return os.path.join(REPO_ROOT, "notes", "day-04", "pdfs", base_name)
        if filename.startswith("Day 3/") or "anaphora" in fname_lower or "coreference" in fname_lower:
            return os.path.join(REPO_ROOT, "notes", "day-03", "pdfs", base_name)
        if "session" in fname_lower or "shallow" in fname_lower or "lab" in fname_lower:
            return os.path.join(REPO_ROOT, "notes", target_day, "pdfs", base_name)
        return os.path.join(REPO_ROOT, "reference-books", base_name)

    # 4. Teacher reference python code (.py) from cloud
    if filename.endswith(".py"):
        if filename.startswith("DAY 4/"):
            return os.path.join(REPO_ROOT, "notes", "day-04", "code", "Day 4", base_name)
        elif filename.startswith("Day 3/"):
            return os.path.join(REPO_ROOT, "notes", "day-03", "code", "Day 3", base_name)
        elif filename.startswith("DAY 2/"):
            return os.path.join(REPO_ROOT, "notes", "day-02", "code", "DAY 2", base_name)
        elif "morphologica" in fname_lower:
            return os.path.join(REPO_ROOT, "notes", "day-02", "code", "DAY 2", base_name)
        return os.path.join(REPO_ROOT, "notes", target_day, "code", base_name)

    # 5. Dataset files (.csv)
    if filename.endswith(".csv"):
        return os.path.join(REPO_ROOT, "notes", target_day, base_name)

    # 6. Zips
    if filename.endswith(".zip"):
        return os.path.join(REPO_ROOT, "raw-downloads", base_name)

    return None


def download_file(rel_path, target_path):
    """Download single file from WebDAV to local target."""
    encoded_rel = urllib.parse.quote(rel_path)
    file_url = CLOUD_BASE_URL + encoded_rel
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    print(f"[Downloading] {rel_path} -> {os.path.relpath(target_path, REPO_ROOT)} ...", end=" ", flush=True)
    r = requests.get(file_url, auth=AUTH, stream=True, timeout=60)
    if r.status_code == 200:
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                f.write(chunk)
        print("[OK]")
        return True
    else:
        print(f"[ERROR Status {r.status_code}]")
        return False


def main():
    print("=" * 70)
    print("[Network] CDAC CloudFoundry Sync Tool")
    print(f"[Remote] {CLOUD_BASE_URL}")
    print("=" * 70)

    print("\n[Search] Checking remote server for files...")
    remote_files = list_remote_files()
    if not remote_files:
        print("No files discovered.")
        return

    print(f"Found {len(remote_files)} item(s) on server.\n")
    downloaded_count = 0

    for item in remote_files:
        dest = determine_destination(item)
        if not dest:
            print(f"  [Skip] Skipping unknown item: {item}")
            continue

        if os.path.exists(dest):
            # Check file size to see if it's partially downloaded
            print(f"  [Up-to-Date] {os.path.relpath(dest, REPO_ROOT)}")
        else:
            print(f"  [New Item] {item}")
            success = download_file(item, dest)
            if success:
                downloaded_count += 1
                if dest.endswith(".zip"):
                    zip_name = os.path.basename(dest)
                    day_match = re.search(r"day[\s_]*(\d+)", zip_name.lower())
                    if day_match:
                        day_num = int(day_match.group(1))
                        day_dir = os.path.join(REPO_ROOT, "notes", f"day-{day_num:02d}")
                        print(f"    [Extract] Unpacking {zip_name} to {day_dir}...")
                        try:
                            with zipfile.ZipFile(dest, "r") as zf:
                                zf.extractall(day_dir)
                            print(f"    [OK] Extracted!")
                        except Exception as e:
                            print(f"    [WARNING] Extract error: {e}")

    print("\n" + "=" * 70)
    if downloaded_count > 0:
        print(f"[Done] Sync Complete! Downloaded and merged {downloaded_count} new item(s).")
    else:
        print("[Done] Sync Complete! All files are up-to-date.")
    print("=" * 70)


if __name__ == "__main__":
    main()
