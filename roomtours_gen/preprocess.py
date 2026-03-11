#!/usr/bin/env python3
"""Run point-cloud preprocessing on Pi3 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.runtime import PI3_ROOT, PREPROCESS_ROOT, ROOT_DIR, print_command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the point-cloud preprocessing stage locally")
    parser.add_argument("--input-root", type=Path, default=PI3_ROOT, help="Directory containing Pi3 outputs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PREPROCESS_ROOT,
        help="Directory where preprocessed point clouds are written",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "pipeline" / "preprocess" / "default_config.json",
        help="Preprocessing config JSON",
    )
    parser.add_argument(
        "--ply-path",
        type=Path,
        action="append",
        default=[],
        help="Process only the specified input PLY file. Can be passed multiple times.",
    )
    parser.add_argument("--max-entries", type=int, default=0, help="Maximum number of PLY files to process (0 = all)")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing preprocessing outputs")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate per-step point clouds in addition to the final output",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs from the preprocessing pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved targets without executing them")
    args = parser.parse_args()

    if args.max_entries < 0:
        parser.error("--max-entries must be >= 0")
    if not args.config.exists():
        parser.error(f"--config not found: {args.config}")

    input_paths = discover_input_plys(
        input_root=args.input_root,
        explicit_paths=args.ply_path,
        max_entries=args.max_entries,
        overwrite_existing=args.overwrite_existing,
        output_root=args.output_root,
        parser=parser,
    )
    if not input_paths:
        print("[INFO] no PLY files selected for preprocessing", flush=True)
        return 0

    print(f"[INFO] selected point clouds: {len(input_paths)}", flush=True)
    for input_path in input_paths:
        output_dir = default_output_dir(input_path, args.input_root, args.output_root)
        command = [
            "preprocess",
            "--input-ply",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--config",
            str(args.config),
        ]
        if args.save_intermediates:
            command.append("--save-intermediates")
        print_command(command)

    if args.dry_run:
        return 0

    return run_preprocess_batch(
        input_paths=input_paths,
        input_root=args.input_root,
        output_root=args.output_root,
        config_path=args.config,
        overwrite_existing=args.overwrite_existing,
        save_intermediates=args.save_intermediates,
        verbose=args.verbose,
    )


def discover_input_plys(
    input_root: Path,
    explicit_paths: list[Path],
    max_entries: int,
    overwrite_existing: bool,
    output_root: Path,
    parser: argparse.ArgumentParser,
) -> list[Path]:
    if explicit_paths:
        paths = []
        for path in explicit_paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                parser.error(f"--ply-path not found: {path}")
            if resolved.suffix.lower() != ".ply":
                parser.error(f"--ply-path must point to a .ply file: {path}")
            paths.append(resolved)
    else:
        root = input_root.expanduser().resolve()
        if not root.exists():
            parser.error(f"--input-root not found: {input_root}")
        paths = sorted(path.resolve() for path in root.rglob("pi3.ply") if path.is_file())

    deduped = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)

    if not overwrite_existing:
        deduped = [
            path
            for path in deduped
            if not (default_output_dir(path, input_root, output_root) / "coord.npy").exists()
        ]

    if max_entries > 0:
        deduped = deduped[:max_entries]
    return deduped


def default_output_dir(input_ply: Path, input_root: Path, output_root: Path) -> Path:
    input_ply = input_ply.expanduser().resolve()
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    try:
        rel_path = input_ply.relative_to(input_root)
        parent = rel_path.parent
        if len(parent.parts) >= 2:
            video_name, scene_name = parent.parts[-2], parent.parts[-1]
            return output_root / f"{video_name}_{scene_name}"
        if parent.parts:
            return output_root / parent.parts[-1]
        return output_root / input_ply.stem
    except ValueError:
        return output_root / input_ply.parent.name


def run_preprocess_batch(
    input_paths: list[Path],
    input_root: Path,
    output_root: Path,
    config_path: Path,
    overwrite_existing: bool,
    save_intermediates: bool,
    verbose: bool,
) -> int:
    from plyfile import PlyData

    from pipeline.preprocess.main_pipeline import PreprocessPipeline

    pipeline = PreprocessPipeline(config_path=str(config_path), verbose=verbose)

    for index, input_path in enumerate(input_paths, start=1):
        output_dir = default_output_dir(input_path, input_root, output_root)
        coord_output_path = output_dir / "coord.npy"
        report_path = output_dir / "report.json"

        if coord_output_path.exists() and not overwrite_existing:
            print(f"[{index}/{len(input_paths)}] skip existing: {coord_output_path}", flush=True)
            continue

        print(f"[{index}/{len(input_paths)}] preprocess {input_path} -> {output_dir}", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        with input_path.open("rb") as handle:
            ply_data = PlyData.read(handle)
        vertex = ply_data["vertex"].data

        coords = stack_vertex_fields(vertex, ("x", "y", "z"), dtype="float32")
        colors = None
        if all(name in vertex.dtype.names for name in ("red", "green", "blue")):
            colors = stack_vertex_fields(vertex, ("red", "green", "blue"), dtype="uint8")
        normals = None
        if all(name in vertex.dtype.names for name in ("nx", "ny", "nz")):
            normals = stack_vertex_fields(vertex, ("nx", "ny", "nz"), dtype="float32")

        coords_final, colors_final, normals_final, report = pipeline.process(
            coords,
            colors,
            normals,
            save_intermediates=save_intermediates,
            output_dir=str(output_dir),
        )
        save_final_npy(output_dir, coords_final, colors_final, normals_final)
        pipeline.save_report(report, report_path)

    return 0


def stack_vertex_fields(vertex, field_names: tuple[str, str, str], dtype: str):
    import numpy as np

    arrays = [vertex[name].astype(dtype) for name in field_names]
    return np.stack(arrays, axis=1)


def save_final_npy(output_dir: Path, coords, colors, normals) -> None:
    import numpy as np

    np.save(output_dir / "coord.npy", coords)
    if colors is not None:
        np.save(output_dir / "color.npy", colors)
    if normals is not None:
        np.save(output_dir / "normal.npy", normals)


if __name__ == "__main__":
    raise SystemExit(main())
