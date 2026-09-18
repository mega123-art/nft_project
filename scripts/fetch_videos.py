"""
Phase 4 step 0 (not in the numbered PLAN.md list, but needed before step 1):
download supplementary clips with yt-dlp to fill the classes our own phone
footage may not cover well -- mainly signal_red/signal_green (zero training
data as of Phase 2/3) and hard conditions like night and rain.

Licence audit trail: every download, successful or not, is not what gets
recorded -- only successful downloads are -- but every successful one is
recorded into data/video_manifest.json with the source url, yt-dlp's video
id, title, uploader, upload date, licence, duration, resolution, the output
filename and a download timestamp. This exists so the final report can cite
where each clip came from, and so the team can prove what was used if a
licence is ever questioned. If yt-dlp reports no licence field at all, that
is recorded explicitly as "unknown" -- never guessed, never omitted, because
a missing manifest entry is indistinguishable from "forgot to check" later.

This script does not enforce Creative-Commons-only. It prints a clear
warning when a downloaded clip's licence is not Creative Commons, but still
downloads it -- the user is the one who decides whether a clip is usable,
this script's job is only to make sure that decision is not made blind.
"""

import argparse
import datetime
import glob
import json
import os

import yt_dlp

DEFAULT_URL_FILE = "data/video_sources.txt"
DEFAULT_OUTPUT_DIR = "data/raw_videos"
DEFAULT_MANIFEST = "data/video_manifest.json"


def _ffmpeg_path():
    """Locate an ffmpeg binary without requiring a system install."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def read_urls_from_file(path):
    """One URL per line, '#' starts a comment (whole-line or trailing)."""
    if not os.path.isfile(path):
        return []

    urls = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                urls.append(line)
    return urls


def load_manifest(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_manifest(path, entries):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def already_downloaded(url, manifest):
    return any(entry["url"] == url for entry in manifest)


def find_downloaded_file(output_dir, video_id):
    """yt-dlp's outtmpl names the file after the video id; find it after the
    download+merge step actually finishes, rather than guessing the
    extension ahead of time (merging can change it)."""
    matches = sorted(glob.glob(os.path.join(output_dir, f"{video_id}.*")))
    if not matches:
        return None
    # prefer .mp4 if more than one file landed (e.g. a leftover .part or
    # a separate audio track that failed to merge)
    for match in matches:
        if match.endswith(".mp4"):
            return match
    return matches[0]


def build_manifest_entry(info, url, output_dir):
    video_id = info.get("id", "unknown")
    width = info.get("width")
    height = info.get("height")
    resolution = f"{width}x{height}" if width and height else "unknown"

    filename = find_downloaded_file(output_dir, video_id)

    return {
        "url": url,
        "video_id": video_id,
        "title": info.get("title", "unknown"),
        "uploader": info.get("uploader", "unknown"),
        "upload_date": info.get("upload_date", "unknown"),
        # yt-dlp gives None when the platform reports no licence at all --
        # record that explicitly rather than leaving the key out.
        "licence": info.get("license") or "unknown",
        "duration_seconds": info.get("duration", "unknown"),
        "resolution": resolution,
        "filename": filename,
        "downloaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def warn_if_not_creative_commons(entry):
    licence = entry["licence"]
    if licence != "unknown" and "creative commons" in licence.lower():
        return
    print(
        f"  WARNING: licence for '{entry['title']}' is '{licence}', not confirmed Creative "
        "Commons. This clip may not be redistributable -- check the source before it goes "
        "into a published dataset."
    )


def download_one(url, output_dir, manifest, force):
    if not force and already_downloaded(url, manifest):
        print(f"skip (already in manifest): {url}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        # Video only, capped at 1080p, and no audio track at all.
        #
        # We extract still frames from these clips; the soundtrack is dead
        # weight. It also avoids a hard dependency: YouTube serves its
        # higher qualities as separate video and audio streams, and asking
        # for both makes yt-dlp merge them with ffmpeg, which is not
        # installed here (same gap that makes extract_frames.py use
        # OpenCV). Requesting a single video-only stream needs no merge.
        # ffmpeg comes from the imageio-ffmpeg wheel rather than the system
        # package, so no root access is needed to merge YouTube's separate
        # video and audio streams. Falls back to whatever is on PATH.
        "ffmpeg_location": _ffmpeg_path(),
        "format": "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[height<=1080][ext=mp4]/best[height<=1080]",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    print(f"downloading: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        # one bad URL (private, deleted, region-blocked, typo'd) must not
        # kill a run that has more URLs queued behind it
        print(f"  FAILED: {e}")
        return None

    entry = build_manifest_entry(info, url, output_dir)
    warn_if_not_creative_commons(entry)
    print(f"  ok: {entry['title']} ({entry['resolution']}, {entry['duration_seconds']}s) -> {entry['filename']}")
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Download supplementary clips with yt-dlp and record a licence audit trail (Phase 4)"
    )
    parser.add_argument("urls", nargs="*", help="video URLs (in addition to any in --url-file)")
    parser.add_argument(
        "--url-file",
        default=DEFAULT_URL_FILE,
        help=f"text file of URLs, one per line, '#' comments allowed (default {DEFAULT_URL_FILE})",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"default {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"default {DEFAULT_MANIFEST}")
    parser.add_argument("--force", action="store_true", help="redownload even if the URL is already in the manifest")
    args = parser.parse_args()

    urls = read_urls_from_file(args.url_file) + list(args.urls)
    if not urls:
        print(f"no URLs given and none found in {args.url_file}")
        raise SystemExit(1)

    manifest = load_manifest(args.manifest)

    downloaded = 0
    failed = 0
    for url in urls:
        entry = download_one(url, args.output_dir, manifest, args.force)
        if entry is None:
            # either skipped (already had it) or failed -- either way there
            # is nothing new to add to the manifest
            if not already_downloaded(url, manifest):
                failed += 1
            continue
        # replace any existing entry for this URL (relevant on --force)
        manifest = [e for e in manifest if e["url"] != url]
        manifest.append(entry)
        save_manifest(args.manifest, manifest)
        downloaded += 1

    print(f"\ndone: {downloaded} downloaded, {failed} failed, {len(manifest)} total entries in {args.manifest}")


if __name__ == "__main__":
    main()
