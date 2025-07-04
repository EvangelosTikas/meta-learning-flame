import os
import tempfile
import pytest
import builtins
import warnings
from unittest import mock
from flameAI.components import data_loaders


COMMON_MSG_TMETA: str = "Skipping because torchmeta is not available"

@pytest.fixture
def temp_bin_file():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp.write(b"mock content")
    tmp.close()
    yield tmp.name
    os.remove(tmp.name)


def test_download_if_url_cached(monkeypatch: pytest.MonkeyPatch, temp_bin_file: builtins.str):
    url = "http://example.com/data"
    data_loaders._CACHE[url] = temp_bin_file
    assert data_loaders.download_if_url(url) == temp_bin_file


def test_download_if_url_new(monkeypatch: pytest.MonkeyPatch):
    url = "http://example.com/data"
    response = mock.Mock()
    response.content = b"abc123"
    response.raise_for_status = mock.Mock()
    with mock.patch("requests.get", return_value=response):
        result_path = data_loaders.download_if_url(url)
        assert os.path.exists(result_path)
        with open(result_path, "rb") as f:
            assert f.read() == b"abc123"
        os.remove(result_path)


def test_download_if_not_url():
    path = "/some/local/file"
    assert data_loaders.download_if_url(path) == path


def test_missing_dataset_torchmeta(monkeypatch: pytest.MonkeyPatch):
    if not data_loaders.HAS_TORCHMETA:
        warnings.warn("torchmeta not installed — skipping test_missing_dataset_torchmeta", UserWarning)
        pytest.skip(COMMON_MSG_TMETA)

    with pytest.raises(ValueError):
        data_loaders.MetaDatasetLoader("nonexistent_dataset", use_torchmeta=True)


def test_fallback_loader_not_implemented():
    with pytest.skip(COMMON_MSG_TMETA):
        data_loaders.MetaDatasetLoader("mnist", use_torchmeta=False)


def test_example_use_dataset_runs(monkeypatch: pytest.MonkeyPatch):
    # Just run to make sure there's no crash
    if data_loaders.HAS_TORCHMETA:
        monkeypatch.setattr(data_loaders, "MetaDatasetLoader", mock.Mock())
        data_loaders.example_use_dataset("omniglot")
    else:
        with pytest.skip(COMMON_MSG_TMETA):
            data_loaders.example_use_dataset("mnist")

# Regression Testing (optional)
# fixture test
def test_cached_url_consistency(temp_bin_file: builtins.str):
    data_loaders._CACHE.clear()
    url = "http://example.com/resource"
    data_loaders._CACHE[url] = temp_bin_file
    result = data_loaders.download_if_url(url)
    assert result == temp_bin_file  # deterministic output


# Test a torchmeta dataset (requires actual download
@pytest.mark.skipif(not data_loaders.HAS_TORCHMETA, reason="torchmeta not installed")
def test_omniglot_loader_consistency():
    loader = data_loaders.MetaDatasetLoader("omniglot", split="val", shots=1)
    dl = loader.get_dataloader()
    task = next(iter(dl))
    assert "train" in task
    assert isinstance(task["train"], list)
