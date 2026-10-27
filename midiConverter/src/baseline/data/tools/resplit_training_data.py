"""One-off migration of TRAINING_PATH to a leak-free split by source recording (review items D1, D2).

1. Own-kit files (``kick3.wav``, ``hi-hat5 2.wav``, ...) are removed from the training data. Byte-identical
   copies of CUTTED_VAL_SETS_PATH files are deleted; the others are moved to ``OWN_KIT_EXTRA`` so the own kit
   stays a test-only domain.
2. The remaining files are grouped by source recording, and whole groups are assigned to train/val/test,
   stratified per class:
   - sample library ``<MIC>_<piece>_<code>.<rr>.wav``: one group per piece and code, so the four microphones
     (DI, IN, MN, ST) and the round robins of one hit stay together
   - IDMT-SMT-Drums ``<kit>_<take>#...``: one group per take
   - anything else: one group per file
   Files with identical audio are always put in the same group.

Run with ``--apply`` to move files; without it the script only prints the plan.
"""
import argparse
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.constants import CUTTED_VAL_SETS_PATH, DRUM_INSTRUMENTS, INSTRUMENTS_DATA_SET_PATH, REPO_ROOT

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
OWN_KIT_EXTRA = REPO_ROOT / "data" / "input" / "ownKitExtra"
SEED = 42

OWN_KIT_PATTERN = re.compile(r"^(kick|snare|hi-hat|crash|ride|rack_tom|floor_tom)\d+( \d+)?\.wav$")
LIBRARY_PATTERN = re.compile(r"^(DI|IN|MN|ST)_(?P<piece>.+)_(?P<code>\d{4})\.\d+\.wav$")
IDMT_PATTERN = re.compile(r"^(?P<take>(RealDrum|WaveDrum|TechnoDrum)\d+_\d+)#")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def source_key(cls: str, name: str) -> str:
    if match := LIBRARY_PATTERN.match(name):
        return f"{cls}/library/{match['piece']}/{match['code']}"
    if match := IDMT_PATTERN.match(name):
        return f"{cls}/idmt/{match['take']}"
    return f"{cls}/file/{name}"


def group_files(files: list[Path]) -> dict[str, list[Path]]:
    """Groups by source key and merges groups that share identical audio (union-find)."""
    parent = {}

    def find(key):
        while parent.setdefault(key, key) != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first_key_of_hash = {}
    keys = {}
    for path in files:
        key = source_key(path.parent.name, path.name)
        keys[path] = key
        digest = md5(path)
        if digest in first_key_of_hash:
            parent[find(key)] = find(first_key_of_hash[digest])
        else:
            first_key_of_hash[digest] = key

    groups = defaultdict(list)
    for path in files:
        groups[find(keys[path])].append(path)
    return groups


def assign_splits(groups: dict[str, list[Path]], rng: random.Random) -> dict[str, str]:
    """Greedy stratified assignment: each group goes to the split furthest below its target size."""
    assignment = {}
    by_class = defaultdict(list)
    for key, paths in groups.items():
        by_class[paths[0].parent.name].append(key)

    for cls, keys in sorted(by_class.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        keys.sort(key=lambda k: -len(groups[k]))  # place big groups first, ties stay shuffled
        total = sum(len(groups[k]) for k in keys)
        filled = Counter()
        for key in keys:
            split = max(SPLITS, key=lambda s: SPLITS[s] * total - filled[s])
            assignment[key] = split
            filled[split] += len(groups[key])
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Move the files (default: dry run).")
    args = parser.parse_args()

    root = Path(INSTRUMENTS_DATA_SET_PATH)
    files = sorted(p for split in SPLITS for p in (root / split).glob("*/*.wav"))
    cutted_hashes = {md5(p) for p in Path(CUTTED_VAL_SETS_PATH).glob("*.wav")}

    own_kit = [p for p in files if OWN_KIT_PATTERN.match(p.name)]
    to_delete = [p for p in own_kit if md5(p) in cutted_hashes]
    to_extra = [p for p in own_kit if p not in to_delete]
    library = [p for p in files if p not in own_kit]

    groups = group_files(library)
    assignment = assign_splits(groups, random.Random(SEED))

    # Byte-identical copies that would land on the same target path are dropped, keeping one.
    moves, duplicates, targets = [], [], {}
    for key, paths in groups.items():
        for path in paths:
            dst = root / assignment[key] / path.parent.name / path.name
            if dst in targets:
                if md5(targets[dst]) != md5(path):
                    raise FileExistsError(f"two different files map to {dst}: {targets[dst]}, {path}")
                duplicates.append(path)
                continue
            targets[dst] = path
            moves.append((path, dst))
    print(f"identical duplicates to delete: {len(duplicates)}")

    print(f"own-kit copies of cuttedValSets to delete: {len(to_delete)}")
    print(f"other own-kit files to move to {OWN_KIT_EXTRA}: {len(to_extra)}")
    print(f"library/IDMT files: {len(library)} in {len(groups)} source groups")
    counts = Counter((dst.parent.name, dst.parent.parent.name) for _, dst in moves)
    print(f"{'class':10s} " + " ".join(f"{s:>6s}" for s in SPLITS))
    for cls in DRUM_INSTRUMENTS:
        print(f"{cls:10s} " + " ".join(f"{counts[(cls, s)]:6d}" for s in SPLITS))

    if not args.apply:
        print("Dry run. Re-run with --apply to move files.")
        return

    for path in to_delete + duplicates:
        path.unlink()
    for path in to_extra:
        target = OWN_KIT_EXTRA / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
    # Two-phase move, so a file never overwrites another one with the same name in the target split.
    staging = root / "_staging"
    for i, (src, _) in enumerate(moves):
        tmp = staging / f"{i:05d}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, tmp)
    for i, (_, dst) in enumerate(moves):
        if dst.exists():
            raise FileExistsError(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging / f"{i:05d}.wav", dst)
    staging.rmdir()
    print("Done.")


if __name__ == "__main__":
    main()
