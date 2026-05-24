import argparse
from typing import List, Optional

from .config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot logo recognition on a video file."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--yolo-weights", required=True, help="Path to YOLO weights.")
    parser.add_argument(
        "--recog-weights",
        required=True,
        help="Path to ArcFace recognition weights.",
    )
    parser.add_argument(
        "--embed-db",
        required=True,
        help="Path to embedding database (pickle).",
    )
    parser.add_argument("--output", required=True, help="Path to output video.")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.7,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--recog-threshold",
        type=float,
        default=0.4,
        help="Recognition similarity threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. cuda:0, cpu).",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = PipelineConfig(
        video_path=args.video,
        yolo_weights=args.yolo_weights,
        recog_weights=args.recog_weights,
        embed_db_path=args.embed_db,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        recog_threshold=args.recog_threshold,
        device=args.device,
    )

    from .pipeline import run_pipeline

    run_pipeline(config)
