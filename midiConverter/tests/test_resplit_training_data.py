import random

from src.baseline.data.tools.resplit_training_data import assign_splits, group_files


def _files(tmp_path):
    specs = [
        ("train", "snare", "DI_Snare_Strike_1111.1.wav", b"a"),
        ("val", "snare", "ST_Snare_Strike_1111.2.wav", b"b"),  # same hit, other mic: same group
        ("test", "snare", "DI_Snare_Strike_1112.1.wav", b"c"),
        ("train", "kick", "RealDrum01_10#KD#train_first_hit.wav", b"d"),
        ("val", "kick", "RealDrum01_11#KD#train_first_hit.wav", b"d"),  # identical audio: merged
        ("test", "kick", "Bass Sample 1.wav", b"e"),
    ]
    paths = []
    for split, cls, name, data in specs:
        path = tmp_path / split / cls / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths.append(path)
    return paths


def test_groups_keep_one_hit_together_and_merge_identical_audio(tmp_path):
    groups = group_files(_files(tmp_path))
    sizes = sorted(len(paths) for paths in groups.values())
    assert sizes == [1, 1, 2, 2]


def test_split_does_not_depend_on_file_order(tmp_path):
    files = _files(tmp_path)
    shuffled = files[:]
    random.Random(1).shuffle(shuffled)
    first = assign_splits(group_files(files), random.Random(42))
    second = assign_splits(group_files(shuffled), random.Random(42))
    assert first == second
