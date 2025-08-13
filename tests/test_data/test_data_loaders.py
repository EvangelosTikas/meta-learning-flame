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


# ---------------
# Common messages

COMMON_MSG: str = "Skipping this..."
TESTING_MSG: str = "[TEST] Info: "


# ---------------
# Testing functions

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



from src.flameAI.components.data_loaders import MetaDatasetLoader, register_dataset, _ARR_TASKSETS

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

        # Also patch Taskset
        with patch("src.flameAI.components.data_loaders.learn2learn.data.Taskset") as mock_taskdataset:
            mock_taskdataset.return_value = MagicMock(spec=DataLoader)
            print(f"{TESTING_MSG} mock_dataset_class.return_value {mock_dataset_class.return_value.sampler.__class__.__name__}")
            yield mock_taskdataset


#
# Mock tests
#

# This way, you test the DataLoader logic without mocking all of Learn2Learn, [] Fail
# @patch("learn2learn.vision.benchmarks._TASKSETS", {
#     "mini-imagenet": lambda **kwargs: MagicMock(
#         train=MagicMock(spec=IterableDataset, task_transforms=[]),
#         validation=MagicMock(spec=IterableDataset, task_transforms=[]),
#         test=MagicMock(spec=IterableDataset, task_transforms=[])
#     )
# })
# def test_l2l_meta_dataloader_taskset_mocking():
#     loader = MetaDatasetLoader(dataset_name="mini-imagenet", shots=5, batch_size=2)
#     assert isinstance(loader.get_dataloader(), DataLoader)

@patch("src.flameAI.components.data_loaders.MetaDatasetLoader._load_l2l_dataset", return_value=(MagicMock(spec=IterableDataset), None))
def test_loader_without_mocking_l2l_all(mock_dataset):
    for name in _ARR_TASKSETS:
        loader = MetaDatasetLoader(dataset_name=name, shots=5, batch_size=2)
        assert isinstance(loader.get_dataloader(), DataLoader)

# Mock test
# def test_shape_correctness_in_l2l_transform_pipeline(mock_l2l_dataset: MagicMock | mock.AsyncMock):
#     loader = MetaDatasetLoader(dataset_name="mini-imagenet", shots=1, test_shots=1, batch_size=2)
#     dl = loader.get_dataloader()
#     assert isinstance(dl, DataLoader)


# -----------
# Actual tests

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


def test_register_dataset_invocation():
    dummy_fn = lambda self: ("dummy_dataset", None)
    register_dataset("dummyset", dummy_fn)
    loader = MetaDatasetLoader(dataset_name="dummyset")
    assert loader.dataset == "dummy_dataset"


def test_non_callable_loader_register_raises():
    with pytest.raises(TypeError):
        register_dataset("not_callable", "not_a_function")


#  This test the get_dataloader() -> _load_l2l_dataset -> _load_from_tasksets (param: split)
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_get_dataloader_taskset_structure(split: builtins.str | builtins.str | builtins.str):
    loader = MetaDatasetLoader(dataset_name="omniglot", batch_size=3, split=split, use_l2l_taskset=True)
    dl = loader.get_dataloader()
    assert isinstance(dl, DataLoader)
    batch = next(iter(dl))

    assert isinstance(batch, (tuple, list)), f"Expected tuple or list, got {type(batch)}"
    assert len(batch) == 2, "Expected (data, label)"

    # Data
    assert hasattr(batch[0], 'shape'), "Data should have shape attribute"
    assert batch[0].shape[0] == 3, f"Expected batch size 3, got {batch[0].shape[0]}"

    # Label
    assert hasattr(batch[1], 'shape') or isinstance(batch[1], int), "Label should be tensor-like or int"


#  This test the get_dataloader() -> _load_torchvision_dataset (param: split)
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_get_dataloader_torchds_structure(split: builtins.str | builtins.str | builtins.str):
    loader = MetaDatasetLoader(dataset_name="mnist", batch_size=3, split=split)
    dl = loader.get_dataloader()
    assert isinstance(dl, DataLoader)
    batch = next(iter(dl))

    assert isinstance(batch, (tuple, list)), f"Expected tuple or list, got {type(batch)}"
    assert len(batch) == 2, "Expected (data, label)"

    # Data
    assert hasattr(batch[0], 'shape'), "Data should have shape attribute"
    assert batch[0].shape[0] == 3, f"Expected batch size 3, got {batch[0].shape[0]}"

    # Label
    assert hasattr(batch[1], 'shape') or isinstance(batch[1], int), "Label should be tensor-like or int"




# -----------
# Transforms

def test_custom_transform_applied():
    from torchvision import datasets
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),         # Minimal processing
                    ]))

    # Step 2: Wrap it in MetaDataset
    meta_mnist = learn2learn.data.MetaDataset(mnist)

    # Step 3: Add meta-learning transforms to form tasks (episodes)
    task_transforms = [
                learn2learn.data.transforms.NWays(loader.dataset, 5),
                learn2learn.data.transforms.KShots(loader.dataset, 2)
    ]
    loader = MetaDatasetLoader(dataset_name="mnist", dataset=meta_mnist)

    loader.custom_transforms = task_transforms
    assert isinstance(trnsf, list), "This is not a transforms list"
    for trnsf in loader.custom_transforms:
        assert isinstance(trnsf, learn2learn.data.transforms.TaskTransform)


# Test the default trasnforms
@pytest.mark.parametrize("name", ['mnist', 'fashionmnist', 'cifar10'])
def test_default_transform_for_taskset(name: builtins.str | builtins.str | builtins.str):
    loader = MetaDatasetLoader(dataset_name=name)
    assert isinstance(loader.transform, list )
