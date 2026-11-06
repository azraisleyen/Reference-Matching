import argparse
import json
from pathlib import Path
import signal
import sys
import threading
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="uav-refmatch")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Environment and hardware report")
    run = commands.add_parser("run", help="Run the shared inference pipeline")
    run.add_argument("--video", required=True)
    run.add_argument("--reference", required=True)
    run.add_argument("--mode", choices=["static", "dynamic"])
    run.add_argument("--config")
    run.add_argument(
        "--max-frames", type=int, help="Explicit short validation limit; recorded in config"
    )
    run.add_argument("--output-root", default="runs")
    run.add_argument("--name", default="")
    for name in ("render", "report"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run", required=True, type=Path)
        if name == "render":
            cmd.add_argument("--video", help="Reselect a moved source (hash checked)")
    ui = commands.add_parser("ui")
    ui.add_argument("--output-root", default="runs")
    ui.add_argument("--config")
    ui.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .doctor import doctor

            print(json.dumps(doctor(), indent=2))
        elif args.command == "run":
            from .service import run_experiment

            cancelled = threading.Event()
            signal.signal(signal.SIGINT, lambda *_: cancelled.set())
            path = run_experiment(
                args.video,
                args.reference,
                load_config(args.config, mode=args.mode, max_frames=args.max_frames),
                args.output_root,
                args.name,
                cancelled.is_set,
                lambda value: print(json.dumps(value, ensure_ascii=False), flush=True),
            )
            print(path)
            return 130 if cancelled.is_set() else 0
        elif args.command == "render":
            from .outputs.video_renderer import render_video

            print(render_video(args.run, args.video))
        elif args.command == "report":
            from .outputs.reporting import report

            print(json.dumps(report(args.run), ensure_ascii=False, indent=2))
        elif args.command == "ui":
            from .ui import launch

            launch(args.output_root, load_config(args.config), args.port)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
