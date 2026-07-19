#!/usr/bin/env python3
"""Create four tiny H.264 MP4 clips and Phase 35 video-QA records."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PHASE20 = ROOT / "scripts" / "phase20_make_vqa_smoke_fixture.py"
OUT = ROOT / "data" / "video_smoke"
VIDEOS = OUT / "videos"
EVAL_OUT = ROOT / "data" / "eval" / "video_qa_smoke"


def load_phase20_module():
    spec = importlib.util.spec_from_file_location("phase20_make_vqa_smoke_fixture", PHASE20)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {PHASE20}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_transition(name: str, first: str, second: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required only to generate the Phase 35 smoke MP4 fixtures")
    path = VIDEOS / f"{name}.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={first}:s=64x64:d=0.5:r=4",
            "-f",
            "lavfi",
            "-i",
            f"color=c={second}:s=64x64:d=0.5:r=4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    return path


def main() -> None:
    phase20 = load_phase20_module()
    phase20.main()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    clips = [
        ("red_to_blue", "red", "blue", "blue"),
        ("green_to_yellow", "green", "yellow", "yellow"),
        ("blue_to_red", "blue", "red", "red"),
        ("yellow_to_green", "yellow", "green", "green"),
    ]
    for name, first, second, _answer in clips:
        make_transition(name, first, second)

    records = [
        {
            "video": f"{name}.mp4",
            "question": "What color is shown at the end?",
            "answer": answer,
        }
        for name, _first, _second, answer in clips
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "video_qa_smoke_4.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    eval_records = [
        {
            "video": f"../../video_smoke/videos/{record['video']}",
            "question": record["question"],
            "answer": record["answer"],
        }
        for record in records[:2]
    ]
    (EVAL_OUT / "data.jsonl").write_text(
        "\n".join(json.dumps(record) for record in eval_records) + "\n", encoding="utf-8"
    )
    print(f"video smoke data: {OUT / 'video_qa_smoke_4.jsonl'}")
    print(f"video smoke eval: {EVAL_OUT / 'data.jsonl'}")
    print(f"video smoke clips: {VIDEOS}")


if __name__ == "__main__":
    main()
