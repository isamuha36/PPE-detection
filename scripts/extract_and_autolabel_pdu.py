import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.pdu_extractor import PDUVideoExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract frames and auto-label PDU CCTV videos")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/raw/pdu_videos",
        help="Directory containing PDU video files (.mp4)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/trained/best.pt",
        help="Path to trained YOLO model weights"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pdu_curated",
        help="Output directory for images, labels, and visualizations"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Extraction interval in seconds (default: 10.0s)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process per video (useful for testing, e.g. 50)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for auto-labeling (default: 0.35)"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable saving annotated visualization images"
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep frames even if no PPE or workers were detected"
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    videos = sorted(list(video_dir.glob("*.mp4")))
    if not videos:
        print(f"[!] No .mp4 files found in {video_dir}")
        sys.exit(1)

    print(f"============================================================")
    print(f"   PDU Industrial CCTV Frame Extraction & Auto-Labeling")
    print(f"============================================================")
    print(f"Found {len(videos)} videos in {video_dir}:")
    for v in videos:
        print(f"  - {v.name} ({v.stat().st_size / (1024**3):.2f} GB)")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Sampling interval: {args.interval}s")
    print(f"Confidence threshold: {args.conf}")
    print(f"Visualizations: {'Disabled' if args.no_vis else 'Enabled'}")
    print(f"Filter empty frames: {'No' if args.keep_empty else 'Yes'}")
    if args.max_frames:
        print(f"Max frames per video: {args.max_frames}")
    print(f"============================================================")

    extractor = PDUVideoExtractor(
        model_path=args.model_path,
        output_dir=args.output_dir,
        conf_threshold=args.conf
    )

    all_stats = []
    total_saved = 0
    total_skipped = 0
    aggregate_classes = {}

    for idx, video_path in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] Starting {video_path.name}...")
        video_prefix = f"pdu_v{idx}"
        stats = extractor.process_video(
            video_path=video_path,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            save_visualizations=not args.no_vis,
            filter_empty=not args.keep_empty,
            video_prefix=video_prefix
        )
        all_stats.append(stats)
        total_saved += stats["saved_frames"]
        total_skipped += stats["empty_frames_skipped"]
        for cls_name, cnt in stats["class_distribution"].items():
            aggregate_classes[cls_name] = aggregate_classes.get(cls_name, 0) + cnt

    # Save summary metadata
    summary = {
        "total_videos_processed": len(videos),
        "sampling_interval_seconds": args.interval,
        "confidence_threshold": args.conf,
        "total_frames_saved": total_saved,
        "total_empty_skipped": total_skipped,
        "aggregate_classes": aggregate_classes,
        "video_breakdown": all_stats
    }

    summary_file = Path(args.output_dir) / "curation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print(f"\n============================================================")
    print(f"   Extraction & Auto-Labeling Completed Successfully!")
    print(f"============================================================")
    print(f"Total curated frames saved: {total_saved}")
    print(f"Empty frames filtered out: {total_skipped}")
    print(f"Class distribution detected:")
    for cls_name, cnt in sorted(aggregate_classes.items(), key=lambda x: -x[1]):
        print(f"  - {cls_name:12}: {cnt} instances")
    print(f"Metadata summary written to: {summary_file}")
    print(f"Visualized frames available in: {Path(args.output_dir) / 'visualized'}")

if __name__ == "__main__":
    main()
