"""Fails when the same audio appears in more than one split, or in a split and an external test set.

Splits are the top-level folders of ``--root`` (e.g. ``train``, ``val``, ``test``). Audio is compared by
the MD5 of the file bytes, so only byte-identical copies are caught; near-duplicates (the same hit from
another microphone, another round robin) must be prevented by splitting by source recording.
"""
import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}


def hash_audio_files(folder: Path) -> dict[str, list[Path]]:
    hashes = defaultdict(list)
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in AUDIO_SUFFIXES:
            hashes[hashlib.md5(path.read_bytes()).hexdigest()].append(path)
    return hashes


def find_leaks(root: Path, external: list[Path] = ()) -> list[str]:
    """Returns one message per leak; an empty list means the splits are clean."""
    split_of = {}
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for digest, paths in hash_audio_files(split_dir).items():
            split_of.setdefault(digest, {})[split_dir.name] = paths

    leaks = []
    for digest, splits in split_of.items():
        if len(splits) > 1:
            files = [str(p) for paths in splits.values() for p in paths]
            leaks.append(f"same audio in splits {sorted(splits)}: {files}")

    for ext in external:
        for digest, paths in hash_audio_files(ext).items():
            if digest in split_of:
                files = [str(p) for split_paths in split_of[digest].values() for p in split_paths]
                leaks.append(f"external file {paths[0]} also in training data: {files}")
    return leaks


def main(argv=None) -> int:
    from src.constants import CUTTED_VAL_SETS_PATH, INSTRUMENTS_DATA_SET_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(INSTRUMENTS_DATA_SET_PATH),
                        help="Folder whose subfolders are the splits (default: TRAINING_PATH).")
    parser.add_argument("--external", type=Path, nargs="*", default=[Path(CUTTED_VAL_SETS_PATH)],
                        help="Test sets that must not overlap the training data (default: CUTTED_VAL_SETS_PATH).")
    args = parser.parse_args(argv)

    leaks = find_leaks(args.root, args.external)
    for leak in leaks:
        print(leak)
    print(f"{len(leaks)} leak(s) found in {args.root}")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
