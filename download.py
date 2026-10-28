
from pathlib import Path
import sys
import yt_dlp

DEFAULT_URL = "https://youtu.be/5WaAWU4ef_s?si=-KbtOKVqlisSwZqd"

def download_video(url: str):
    output_dir = Path("downloads")
    output_dir.mkdir(parents=True, exist_ok=True)

    options = {
        # Video-only, maximum 1080p.
        # Prefer MP4, but allow WebM if it offers better quality.
        "format": (
            "bestvideo[height<=1080][ext=mp4]/"
            "bestvideo[height<=1080]/"
            "best[height<=1080][acodec=none]"
        ),
        "outtmpl": str(output_dir / "%(title).180B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "continuedl": True,
        "retries": 10,
        "fragment_retries": 10,
        "ignoreerrors": False,
        "overwrites": False,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if info:
            print("\nDownload completed!")
            print("Title:", info.get("title"))
            print("Resolution:", info.get("resolution", "Unknown"))
            print("Output folder:", output_dir.resolve())

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    download_video(url)
