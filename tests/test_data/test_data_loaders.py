import os
import tempfile
import pytest
import builtins
from unittest import mock
from src.flameAI.components import data_loaders
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from unittest.mock import patch, MagicMock
import torch
import os
import learn2learn



COMMON_MSG: str = "Skipping this..."
TESTING_MSG: str = "[TEST] Info: "
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


from src.flameAI.components.data_loaders import MetaDatasetLoader, register_dataset

@pytest.fixture
def mock_l2l_dataset():
    # Patch where it's *used*, not where it's defined
    with patch("src.flameAI.components.data_loaders.learn2learn.vision.datasets") as mock_datasets:
        # Provide a mock class that would be returned for MiniImageNet
        mock_dataset_class = MagicMock()
        mock_datasets.MiniImageNet = mock_dataset_class
        mock_dataset_instance = MagicMock()
        mock_dataset_class.return_value = mock_dataset_instance
        print(f"{TESTING_MSG} isinstance(dataset, IterableDataset): {isinstance(mock_dataset_class.return_value, IterableDataset)}")
        print(f"{TESTING_MSG} mock_dataset_instance {mock_dataset_class.return_value}, mock_dataset_instance.sampler {type(mock_dataset_class.return_value.sampler)}\n")

        # Also patch TaskDataset
        with patch("src.flameAI.components.data_loaders.learn2learn.data.TaskDataset") as mock_taskdataset:
            mock_taskdataset.return_value = MagicMock(spec=DataLoader)
            print(f"{TESTING_MSG} mock_dataset_class.return_value {mock_dataset_class.return_value.sampler.__class__.__name__}")
            yield mock_taskdataset


def test_l2l_dataset_mocking(mock_l2l_dataset):
    loader = MetaDatasetLoader(dataset_name="miniimagenet", shots=5, batch_size=2)
    assert isinstance(loader.get_dataloader(), DataLoader)


def test_torchvision_dataset():
    loader = MetaDatasetLoader(dataset_name="mnist", shots=1, batch_size=4)
    dl = loader.get_dataloader()
    batch = next(iter(dl))
    x, y = batch
    assert x.shape[0] == 4
    assert y.shape[0] == 4


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        MetaDatasetLoader(dataset_name="nonexistent")


def test_custom_transform_applied():
    custom_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])
    loader = MetaDatasetLoader(dataset_name="mnist", custom_transforms=custom_transform)
    assert loader.transform == custom_transform


def test_register_dataset_invocation():
    dummy_fn = lambda self: ("dummy_dataset", None)
    register_dataset("dummyset", dummy_fn)
    loader = MetaDatasetLoader(dataset_name="dummyset")
    assert loader.dataset == "dummy_dataset"


def test_non_callable_loader_register_raises():
    with pytest.raises(TypeError):
        register_dataset("not_callable", "not_a_function")


def test_get_dataloader_structure():
    loader = MetaDatasetLoader(dataset_name="mnist", batch_size=3)
    dl = loader.get_dataloader()
    batch = next(iter(dl))
    assert isinstance(batch, (tuple, list)), f"Expected tuple or list, got {type(batch)}"
    assert len(batch) == 2
    assert hasattr(batch[0], 'shape')  # check tensor-like


def test_default_transform_for_mnist():
    loader = MetaDatasetLoader(dataset_name="mnist")
    assert isinstance(loader.transform, transforms.Compose)


def test_shape_correctness_in_l2l_transform_pipeline(mock_l2l_dataset):
    loader = MetaDatasetLoader(dataset_name="miniimagenet", shots=1, test_shots=1, batch_size=2)
    dl = loader.get_dataloader()
    assert isinstance(dl, DataLoader)
