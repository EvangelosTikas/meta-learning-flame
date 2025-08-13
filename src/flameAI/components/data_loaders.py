import os
import requests
import tempfile
# import warnings

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import learn2learn
from learn2learn.vision.benchmarks import get_tasksets, list_tasksets
from learn2learn.data import Taskset
from typing import Union, Dict, List, Optional, Tuple, Any

# TaskDataset is deprecated as a structure
# Optional: Dict registry for custom datasets
CUSTOM_LOADERS = {}
CUSTOM_NAMES   = {}
_CACHE = {}

# L2L taskset
_ARR_TASKSETS = [
    'omniglot',
    'mini-imagenet',
    'tiered-imagenet',
    'fc100',
    'cifarfs',
]


"""
Each task (episode) is composed of a small support set and a query set.

You often need to re-apply transforms dynamically per task, not statically once on the whole dataset.

Using standard torchvision.transforms like transforms.ToTensor() or RandomCrop() on the whole dataset before wrapping it in MetaDataset or Taskset will apply the transformation once and not vary across tasks.
"""


class MetaDatasetLoader:
    """Function tree:
    init->
        <- _resolve_dataset
                            -> _load_l2l_dataset-> _load_from_tasksets
                                                -> _load_manual_dataset
                                                                        -> _load_torchvision_dataset
        <- _load_torchvision_dataset

        __init__() : creates the name, root folder, and meta parameters for the dataset/dataloader
        """
    def __init__(self, dataset_name,
                 root='data',
                 dataset=None,
                 ways=5,
                 shots=1,
                 test_shots=15,
                 split='train',
                 batch_size=16,
                 num_tasks=-1,
                 shuffle=False,
                 download=True,
                 use_l2l_taskset=True,
                 custom_transforms=None):

        self.dataset_name = dataset_name.lower()
        self.root = root
        self.dataset = dataset
        self.ways = ways
        self.shots = shots
        self.test_shots = test_shots
        self.split = split
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.shuffle = shuffle
        self.use_l2l_taskset = use_l2l_taskset
        self.download = download
        self.custom_transforms = custom_transforms # NOTE: l2l transforms only


        self.dataset, self.transform = self._resolve_dataset()
        self.dataloader = self._create_dataloader()

    def _resolve_dataset(self):
        # Check learn2learn
        if self.dataset_name in _ARR_TASKSETS:
            return self._load_l2l_dataset()

        # Torchvision fallback (not few-shot aware) : TODO include cases for that
        elif self.dataset_name in ['mnist', 'fashionmnist', 'cifar10']:
            return self._load_torchvision_dataset()

        # Custom loaders
        elif self.dataset_name in CUSTOM_LOADERS:
            return CUSTOM_LOADERS[self.dataset_name](self)

        raise ValueError(f"Unknown dataset: {self.dataset_name}, either register it or use one of the l2l sets:\n \
                        {_ARR_TASKSETS}")

    def _default_transform(self, dataset_name) -> transforms.Compose:
        if dataset_name == 'omniglot':
            return transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor()
            ])
        elif dataset_name == 'miniimagenet':
            return transforms.Compose([
                transforms.Resize(84),
                transforms.ToTensor()
            ])
        elif dataset_name == 'mnist':
            return transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor()
            ])
        else:
            return transforms.ToTensor()

    # Called in _resolve_dataset, during __init__: load a preset of l2l Datasets
    def _load_l2l_dataset(self) -> Tuple[Any, Any]:
        """Load dataset either from predefined tasksets or manual config."""
        if self.use_l2l_taskset:
            return self._load_from_tasksets()
        else:
            return self._load_manual_dataset()

    def _load_from_tasksets(self) -> Tuple[Taskset, List] :
        """Load dataset from learn2learn's predefined tasksets."""
        tasksets = get_tasksets(
            self.dataset_name,
            train_ways=self.ways,
            train_samples=self.shots + self.test_shots,
            test_ways=self.ways,
            test_samples=self.shots + self.test_shots,
            root=self.root
        )
        if self.split == 'train':
            return tasksets.train, tasksets.train.task_transforms
        elif self.split == 'validation':
            return tasksets.validation, tasksets.validation.task_transforms
        elif self.split == 'test':
            return tasksets.test, tasksets.train.task_transforms
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _load_manual_dataset(self) -> Tuple[Any, Any]:
        """Manually configure dataset and transforms."""
        dataset_cls = {
            "omniglot": learn2learn.vision.datasets.FullOmniglot,
            "miniimagenet": learn2learn.vision.datasets.MiniImagenet,
            "fc100": learn2learn.vision.datasets.FC100,
            "cifarfs": learn2learn.vision.datasets.CIFARFS,
            "tieredimagenet": learn2learn.vision.datasets.TieredImagenet
        }.get(self.dataset_name)

        transform = self.custom_transforms or self._default_transform(self.dataset_name)
        base = dataset_cls(root=self.root, download=self.download, transform=transform)
        base = learn2learn.data.MetaDataset(base)

        meta_dataset = learn2learn.data.Taskset(
            base,
            task_transforms=self._meta_transforms(base),
            num_tasks=self.num_tasks
        )
        return meta_dataset, transform

    # Called in _resolve_dataset, during __init__: load a preset of torchvision Datasets
    def _load_torchvision_dataset(self):
        dataset_cls = {
            "mnist": datasets.MNIST,
            "fashionmnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
        }.get(self.dataset_name)

        transform = self._default_transform(self.dataset_name)
        train = self.split == "train"
        dataset = dataset_cls(self.root, train=train, transform=transform, download=self.download)

        meta_ds = learn2learn.data.MetaDataset(dataset)
        # Add meta-learning transforms to form tasks (episodes)
        task_transforms = self._meta_transforms(meta_ds)

        task_dataset = learn2learn.data.Taskset(
            meta_ds,
            task_transforms=task_transforms,
            num_tasks=1000,  # Number of episodes/tasks you want
        )

        return task_dataset, task_transforms


    def _meta_transforms(self, ds: Taskset):
        return  [
            learn2learn.data.transforms.NWays(ds, n=self.ways),
            learn2learn.data.transforms.KShots(ds, k=self.shots + self.test_shots),
            learn2learn.data.transforms.LoadData(ds),
            learn2learn.data.transforms.RemapLabels(ds),
            learn2learn.data.transforms.ConsecutiveLabels(ds),
        ]

    def _create_dataloader(self):
        # simply return a created Dataloader object from dataset
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)


    def get_dataloader(self):
        return self.dataloader


def register_dataset(name: str, loader_fn):
    """Register a custom dataloader"""
    if not callable(loader_fn):
        raise TypeError("loader_fn must be callable")

    CUSTOM_LOADERS[name.lower()] = loader_fn


# Example usage
if __name__ == "__main__":
    loader = MetaDatasetLoader(
        dataset_name="mini-imagenet",
        shots=5,
        batch_size=4,
        split="train"
    )
    print(f"MetaDatasetLoader obj, type : {loader, type(loader)}")

    example_man_loader = loader._load_manual_dataset
    # loader_load_l2l_dataset ->
    for task in loader.get_dataloader():
        # support_x, support_y = task[0]
        # query_x, query_y = task[1]
        print(type(task), len(task))
        if isinstance(task, dict):  # learn2learn task
            x, y = task['data'], task['labels']
        else:  # torchvision fallback
            x, y = task
        print("Data:", x.shape, "Labels:", y.shape)
        break
