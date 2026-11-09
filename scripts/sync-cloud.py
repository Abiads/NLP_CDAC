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
    resp = requests.request("PROPFIND", CLOUD_BASE_URL, auth=AUTH, headers=headers, timeout=20)
    if resp.status_code not in (200, 207):
        print(f"❌ Failed to connect to CloudFoundry: Status {resp.status_code}")
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
    - Session1 / Session2 -> notes/day-01/slides
    - Session3 / Session4 -> notes/day-02/slides
    - DAY 1.zip -> raw-downloads/ & extract to notes/day-01
    - DAY 2.zip -> raw-downloads/ & extract to notes/day-02
    - Assignment -> assignments/
    """
    fname_lower = filename.lower()

    # Session slides (.pptx)
    if filename.endswith(".pptx"):
        if any(s in fname_lower for s in ["session1", "session2"]):
            return os.path.join(REPO_ROOT, "notes", "day-01", "slides", filename)
        elif any(s in fname_lower for s in ["session3", "session4"]):
            return os.path.join(REPO_ROOT, "notes", "day-02", "slides", filename)
        elif any(s in fname_lower for s in ["session5", "session6"]):
            return os.path.join(REPO_ROOT, "notes", "day-03", "slides", filename)
        else:
            return os.path.join(REPO_ROOT, "notes", "day-02", "slides", filename)

    # Assignments (.docx, .pdf)
    if "assignment" in fname_lower:
        day_match = re.search(r"day\s*(\d+)", fname_lower)
        day_num = int(day_match.group(1)) if day_match else 1
        assign_dir = os.path.join(REPO_ROOT, "assignments", f"assignment-{day_num:02d}")
        os.makedirs(assign_dir, exist_ok=True)
        return os.path.join(assign_dir, os.path.basename(filename))

    # Python code (.py)
    if filename.endswith(".py"):
        # Put into day-02/code if it's tokenization or newer, else day-01
        return os.path.join(REPO_ROOT, "notes", "day-02", "code", filename)

    # Day Zips
    if filename.endswith(".zip"):
        return os.path.join(REPO_ROOT, "raw-downloads", os.path.basename(filename))

    # Reference books (.pdf)
    if filename.endswith(".pdf"):
        return os.path.join(REPO_ROOT, "reference-books", os.path.basename(filename))

    return None


def download_file(rel_path, target_path):
    """Download single file from WebDAV to local target."""
    encoded_rel = urllib.parse.quote(rel_path)
    file_url = CLOUD_BASE_URL + encoded_rel
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    print(f"⬇️  Downloading: {rel_path} ...", end=" ", flush=True)
    r = requests.get(file_url, auth=AUTH, stream=True, timeout=30)
    if r.status_code == 200:
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)
        print("✅ Done!")
        return True
    else:
        print(f"❌ Failed (Status {r.status_code})")
        return False


def main():
    print("=" * 65)
    print("🌐 CDAC CloudFoundry Sync Tool")
    print(f"📡 Remote: {CLOUD_BASE_URL}")
    print("=" * 65)

    print("\n🔍 Checking remote server for files...")
    remote_files = list_remote_files()
    if not remote_files:
        print("No files discovered.")
        return

    print(f"Found {len(remote_files)} item(s) on server:\n")
    downloaded_count = 0

    for item in remote_files:
        dest = determine_destination(item)
        if not dest:
            print(f"  ⏭️  Skipping unknown item: {item}")
            continue

        if os.path.exists(dest):
            print(f"  ✔️  Already up-to-date: {os.path.basename(dest)}")
        else:
            print(f"  ✨ New file found: {item}")
            success = download_file(item, dest)
            if success:
                downloaded_count += 1
                # If it's a zip file, optionally extract
                if dest.endswith(".zip"):
                    zip_name = os.path.basename(dest)
                    day_match = re.search(r"day\s*(\d+)", zip_name.lower())
                    if day_match:
                        day_num = int(day_match.group(1))
                        day_dir = os.path.join(REPO_ROOT, "notes", f"day-{day_num:02d}")
                        print(f"     📦 Extracting {zip_name} to {day_dir}...")
                        try:
                            with zipfile.ZipFile(dest, "r") as zf:
                                zf.extractall(day_dir)
                            print(f"     ✅ Extracted!")
                        except Exception as e:
                            print(f"     ⚠️ Extract error: {e}")

    print("\n" + "=" * 65)
    if downloaded_count > 0:
        print(f"🎉 Sync Complete! Downloaded {downloaded_count} new item(s).")
    else:
        print("🎉 Sync Complete! Everything is already up-to-date.")
    print("=" * 65)


if __name__ == "__main__":
    main()

