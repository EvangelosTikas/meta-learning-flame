import os
import tempfile
import pytest
import builtins
from unittest import mock
from flameAI.components import data_loaders
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os


COMMON_MSG_TMETA: str = "Skipping this..."

@pytest.fixture
def temp_bin_file():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp.write(b"mock content")
    tmp.close()
    yield tmp.name
    os.remove(tmp.name)


# def test_download_if_url_cached(monkeypatch: pytest.MonkeyPatch, temp_bin_file: builtins.str):
#     url = "http://example.com/data"
#     data_loaders._CACHE[url] = temp_bin_file
#     assert data_loaders.download_if_url(url) == temp_bin_file


# def test_download_if_url_new(monkeypatch: pytest.MonkeyPatch):
#     url = "http://example.com/data"
#     response = mock.Mock()
#     response.content = b"abc123"
#     response.raise_for_status = mock.Mock()
#     with mock.patch("requests.get", return_value=response):
#         result_path = data_loaders.download_if_url(url)
#         assert os.path.exists(result_path)
#         with open(result_path, "rb") as f:
#             assert f.read() == b"abc123"
#         os.remove(result_path)


# def test_download_if_not_url():
#     path = "/some/local/file"
#     assert data_loaders.download_if_url(path) == path


@pytest.mark.parametrize("dataset_name", ["omniglot", "miniimagenet"])
def test_l2l_dataset_loading(dataset_name):
    loader = MetaDatasetLoader(
        dataset_name=dataset_name,
        shots=1,
        test_shots=1,
        batch_size=2,
        split="train"
    )
    dataloader = loader.get_dataloader()
    batch = next(iter(dataloader))

    assert isinstance(batch, dict), "L2L datasets must return a dict"
    assert "data" in batch and "labels" in batch
    assert batch["data"].shape[0] == 2, "Batch size mismatch"
    assert batch["data"].ndim == 4, "Expected (B, C, H, W) shape"


@pytest.mark.parametrize("dataset_name", ["mnist", "fashionmnist"])
def test_torchvision_dataset_loading(dataset_name):
    loader = MetaDatasetLoader(
        dataset_name=dataset_name,
        split="train",
        batch_size=4
    )
    dataloader = loader.get_dataloader()
    batch = next(iter(dataloader))
    x, y = batch
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 4
    assert y.shape[0] == 4


def test_custom_transform():
    custom_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])
    loader = MetaDatasetLoader(
        dataset_name="mnist",
        custom_transforms=custom_tf,
        batch_size=2
    )
    x, _ = next(iter(loader.get_dataloader()))
    assert x.shape[-1] == 64, "Custom transform Resize(64) not applied"


def test_custom_dataset_registration(tmp_path):
    class DummyDataset(Dataset):
        def __init__(self):
            self.data = [torch.randn(3, 32, 32) for _ in range(10)]
            self.labels = list(range(10))

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

        def __len__(self):
            return len(self.data)

    def dummy_loader_fn(meta_loader):
        return DummyDataset(), transforms.ToTensor()

    register_dataset("dummy", dummy_loader_fn)

    loader = MetaDatasetLoader("dummy", batch_size=4)
    x, y = next(iter(loader.get_dataloader()))
    assert x.shape == (4, 3, 32, 32)
    assert isinstance(y, torch.Tensor)


def test_invalid_dataset_name():
    with pytest.raises(ValueError):
        MetaDatasetLoader("nonexistent_dataset")
