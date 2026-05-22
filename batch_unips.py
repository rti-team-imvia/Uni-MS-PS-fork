"""
Batch RTI Acquisition Processor for Uni-MS-PS
----------------------------------------------
Recursively finds every sub-folder containing a .lp file under INPUT_ROOT,
stages the images into a temporary folder, loads the Uni-MS-PS model ONCE,
runs it for every case, and saves normal_uni.png into the shared output tree
alongside outputs from other models.

The .lp files are used only for discovery (finding which folders contain RTI
data). Light directions are NOT used – the model runs in uncalibrated mode.

Input structure assumed:
  palermo_2026/
    CHEMI_IT_0001/
      CHEMI_IT_0001_01/
        images/
          Face_A/
            rti/
              IMG_0001.JPG ...
              LP_45_GD.lp

Output structure (strips 'images/', 'Face_A/', and the leaf rti/ folder):
  palermo_2026_output/
    CHEMI_IT_0001/
      CHEMI_IT_0001_01/
        normal_uni.png

Usage
-----
  cd D:/cheminova/myCODE/Uni-MS-PS-fork
  python batch_unips.py
  python batch_unips.py --input_root "D:/cheminova/rti-acquisitions/palermo_2026" --output_root "D:/cheminova/rti-acquisitions/palermo_2026_output"
  python batch_unips.py --input_root "D:/cheminova/rti-acquisitions/palermo_2026" --cuda --nb_img 10
"""

from __future__ import print_function, division

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

# Make repo root importable
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

import os

import torch

from utils import load_model
from run import run as unips_run

# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}
MASK_FILENAME = 'mask.png'
RGB_FILENAME = 'RGB.jpg'
NORMAL_FILENAME = 'normal_uni.png'
# Components to drop when building the output path (exact names, lower-cased)
_PATH_PARTS_TO_DROP = {'images'}
# Prefixes of components to drop (lower-cased)
_PATH_PREFIXES_TO_DROP = {'face'}
# ---------------------------------------------------------------------------


def find_lp_case_dirs(input_root: Path) -> list:
    """Return every unique folder containing at least one .lp file, sorted."""
    case_dirs = set()
    for path in input_root.rglob('*.lp'):
        if path.is_file():
            case_dirs.add(path.parent)
    return sorted(case_dirs)


def find_rti_images(case_dir: Path) -> list:
    """Return sorted image files in case_dir, excluding mask / RGB / normals."""
    skip = {MASK_FILENAME, RGB_FILENAME, NORMAL_FILENAME}
    return [
        p for p in sorted(case_dir.iterdir())
        if p.is_file()
        and p.name not in skip
        and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def stage_case(case_dir: Path, staging_dir: Path) -> Path:
    """
    Copy image files from case_dir into staging_dir.
    No mask is created – load_imgs_mask auto-creates an all-ones mask when
    mask.png is absent, which is what we want for uncalibrated inference.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    images = find_rti_images(case_dir)
    if not images:
        raise FileNotFoundError(f'No image files in {case_dir}')
    for img in images:
        shutil.copy2(img, staging_dir / img.name)
    return images[0]


def _should_drop_part(part: str) -> bool:
    lo = part.lower()
    if lo in _PATH_PARTS_TO_DROP:
        return True
    return any(lo.startswith(pfx) for pfx in _PATH_PREFIXES_TO_DROP)


def compute_output_relative(relative_dir: Path) -> Path:
    """
    Strip generic wrapper folders and the leaf RTI folder from the relative path.

    CHEMI_IT_0001/CHEMI_IT_0001_01/images/Face_A/rti
    -> CHEMI_IT_0001/CHEMI_IT_0001_01
    """
    parts = [p for p in relative_dir.parts if not _should_drop_part(p)]
    parts = parts[:-1]  # drop the leaf rti/ folder
    return Path(*parts) if parts else Path('.')


def run_batch(args: argparse.Namespace) -> None:
    input_root: Path = args.input_root.resolve()
    output_root: Path = args.output_root.resolve()
    weights_path: str = str(args.weights_path)

    case_dirs = find_lp_case_dirs(input_root)
    if not case_dirs:
        sys.exit(f'ERROR: No .lp files found under {input_root}')

    print('================================================================')
    print(f'  Batch Uni-MS-PS run  –  {len(case_dirs)} cases found')
    print(f'  Input  : {input_root}')
    print(f'  Output : {output_root}')
    print('================================================================\n')

    # ------------------------------------------------------------------
    # Load the model ONCE for all cases
    # ------------------------------------------------------------------
    device_str = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print(f'Using device: {device_str}\n')

    model = load_model(
        path_weight=weights_path,
        cuda=args.cuda,
        calibrated=False,
        mode_inference=True,
        batch_size_encoder=args.batch_encoder,
        batch_size_transformer=args.batch_transformer,
    )

    total_start = time.time()
    skipped = 0

    for index, case_dir in enumerate(case_dirs, start=1):
        relative_dir = case_dir.relative_to(input_root)
        out_relative = compute_output_relative(relative_dir)
        final_output_dir = output_root / out_relative

        print(f'[{index}/{len(case_dirs)}] {relative_dir}')
        print(f'            -> {out_relative}')

        # Skip if already completed (use --force to reprocess)
        if not args.force and (final_output_dir / NORMAL_FILENAME).exists():
            print('  ALREADY DONE – skipping\n')
            skipped += 1
            continue

        t0 = time.time()

        try:
            if not find_rti_images(case_dir):
                print('  SKIPPED (no images)\n')
                skipped += 1
                continue

            # Run Uni-MS-PS directly on the source folder.
            # load_imgs_mask already ignores non-image files (.lp, .txt, etc.)
            # so no staging / file-copy step is needed.
            final_output_dir.mkdir(parents=True, exist_ok=True)
            unips_run(
                model=model,
                path_obj=str(case_dir),
                nb_img=args.nb_img,
                folder_save=str(final_output_dir),
                obj_name='normal_uni',
                calibrated=False,
                max_size=args.max_size if args.max_size > 0 else None,
            )

            # Remove the .mat sidecar – we only want the PNG
            mat_file = final_output_dir / 'normal_uni.mat'
            if mat_file.exists():
                mat_file.unlink()

        except Exception:
            print('  SKIPPED (model error):')
            traceback.print_exc()
            skipped += 1
            continue

        elapsed = time.time() - t0
        print(f'  Done  ->  {final_output_dir}  ({elapsed:.1f} s)\n')

        # Free GPU memory before next case
        if args.cuda:
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    done = len(case_dirs) - skipped
    print(f'\nFinished: {done} done, {skipped} skipped  ({total_elapsed:.1f} s total)')
    print(f'Outputs: {output_root}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Batch-run Uni-MS-PS (uncalibrated) on RTI acquisition folders.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input_root', type=Path,
                   default=Path('D:/cheminova/rti-acquisitions/palermo_2026'),
                   help='Root folder containing RTI acquisition sub-folders.')
    p.add_argument('--output_root', type=Path,
                   default=Path('D:/cheminova/rti-acquisitions/palermo_2026_output'),
                   help='Output folder (shared with other model outputs).')
    p.add_argument('--weights_path', type=Path, default=Path('weights'),
                   help='Folder containing model_uncalibrated.pth.')
    p.add_argument('--cuda', action='store_true',
                   help='Run on GPU (requires CUDA).')
    p.add_argument('--nb_img', type=int, default=-1,
                   help='Max images per case. -1 = use all available images.')
    p.add_argument('--batch_encoder', type=int, default=3,
                   help='Encoder batch size. Reduce if OOM.')
    p.add_argument('--batch_transformer', type=int, default=5000,
                   help='Transformer batch size. Reduce if OOM.')
    p.add_argument('--max_size', type=int, default=2048,
                   help='Downsample input images to this square size before inference. '
                        'Reduces stages and patches dramatically (2048 = ~18x faster than 8192). '
                        '-1 to disable (use native image size).')
    p.add_argument('--force', action='store_true',
                   help='Re-process cases even if normal_uni.png already exists.')
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    run_batch(args)
