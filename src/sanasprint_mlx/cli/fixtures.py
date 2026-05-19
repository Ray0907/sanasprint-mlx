from __future__ import annotations

import argparse
from pathlib import Path

from sanasprint_mlx.fixtures.reference import generate_reference_fixture
from sanasprint_mlx.fixtures.synthetic import MODEL_REPO, generate_synthetic_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanasprint-mlx-fixtures")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="write a no-download synthetic fixture")
    synthetic.add_argument("--output-dir", required=True, type=Path)
    synthetic.add_argument("--seed", required=True, type=int)
    synthetic.add_argument("--height", type=int, default=8)
    synthetic.add_argument("--width", type=int, default=8)
    synthetic.add_argument("--num-inference-steps", type=int, default=2)
    synthetic.add_argument("--prompt", default="synthetic prompt")

    reference = subparsers.add_parser("reference", help="write an opt-in Diffusers reference fixture")
    reference.add_argument("--output-dir", required=True, type=Path)
    reference.add_argument("--model-repo", default=MODEL_REPO)
    reference.add_argument("--revision", required=True)
    reference.add_argument("--allow-download", action="store_true")
    reference.add_argument("--seed", type=int, default=7)
    reference.add_argument("--height", type=int, default=1024)
    reference.add_argument("--width", type=int, default=1024)
    reference.add_argument("--num-inference-steps", type=int, default=2)
    reference.add_argument("--prompt", default="synthetic prompt")
    reference.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    reference.add_argument("--diffusers-commit")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "synthetic":
        manifest_path = generate_synthetic_fixture(
            args.output_dir,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            prompt=args.prompt,
        )
        print(f"wrote synthetic fixture: {manifest_path}")
        return 0

    if args.command == "reference":
        if not args.allow_download:
            parser.error("reference fixture generation requires --allow-download")
        manifest_path = generate_reference_fixture(
            args.output_dir,
            model_repo=args.model_repo,
            revision=args.revision,
            allow_download=args.allow_download,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            prompt=args.prompt,
            torch_dtype=args.torch_dtype,
            diffusers_commit=args.diffusers_commit,
        )
        print(f"wrote reference fixture: {manifest_path}")
        return 0

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
