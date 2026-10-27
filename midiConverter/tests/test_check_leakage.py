from src.data.check_leakage import find_leaks


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_clean_splits_have_no_leaks(tmp_path):
    _write(tmp_path / "train" / "kick" / "a.wav", b"a")
    _write(tmp_path / "test" / "kick" / "b.wav", b"b")
    assert find_leaks(tmp_path) == []


def test_identical_audio_in_two_splits_is_a_leak(tmp_path):
    _write(tmp_path / "train" / "kick" / "a.wav", b"same")
    _write(tmp_path / "val" / "kick" / "renamed.wav", b"same")
    assert len(find_leaks(tmp_path)) == 1


def test_external_test_file_in_training_data_is_a_leak(tmp_path):
    _write(tmp_path / "data" / "train" / "kick" / "a.wav", b"same")
    _write(tmp_path / "external" / "kick1.wav", b"same")
    assert len(find_leaks(tmp_path / "data", [tmp_path / "external"])) == 1
