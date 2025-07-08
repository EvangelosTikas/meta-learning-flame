import os
import requests
import tempfile
import warnings

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import learn2learn as l2l

# Optional: Dict registry for custom datasets
CUSTOM_LOADERS = {}

_CACHE = {}

def download_if_url(path_or_url):
    if path_or_url.startswith("http"):
        if path_or_url in _CACHE:
            return _CACHE[path_or_url]

        response = requests.get(path_or_url)
        response.raise_for_status()
        suffix = os.path.splitext(path_or_url)[-1] or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()

        _CACHE[path_or_url] = tmp.name
        return tmp.name
    return path_or_url

class MetaDatasetLoader:
    def __init__(self, dataset_name,
                 root='data',
                 ways=5,
                 shots=1,
                 test_shots=15,
                 split='train',
                 batch_size=16,
                 shuffle=True,
                 download=True,
                 custom_transforms=None):

        self.dataset_name = dataset_name.lower()
        self.root = root
        self.ways = ways
        self.shots = shots
        self.test_shots = test_shots
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.download = download
        self.custom_transforms = custom_transforms

        self.dataset, self.transform = self._resolve_dataset()
        self.dataloader = self._create_dataloader()

    def _resolve_dataset(self):
        # Check learn2learn
        if self.dataset_name in ['omniglot', 'miniimagenet', 'fc100', 'cifarfs', 'tieredimagenet']:
            return self._load_l2l_dataset()

        # Torchvision fallback (not few-shot aware) : TODO include cases for that
        elif self.dataset_name in ['mnist', 'fashionmnist', 'cifar10']:
            return self._load_torchvision_dataset()

        # Custom loaders
        elif self.dataset_name in CUSTOM_LOADERS:
            return CUSTOM_LOADERS[self.dataset_name](self)

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _default_transform(self, dataset_name):
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

    def _load_l2l_dataset(self):
        dataset_cls = {
            "omniglot": l2l.vision.datasets.FullOmniglot,
            "miniimagenet": l2l.vision.datasets.MiniImageNet,
            "fc100": l2l.vision.datasets.FC100,
            "cifarfs": l2l.vision.datasets.CIFARFS,
            "tieredimagenet": l2l.vision.datasets.TieredImageNet
        }.get(self.dataset_name)

        transform = self.custom_transforms or self._default_transform(self.dataset_name)
        base = dataset_cls(root=self.root, download=self.download, transform=transform)

        meta_dataset = l2l.data.TaskDataset(
            base,
            task_transforms=[
                l2l.data.transforms.NWays(base, self.ways),
                l2l.data.transforms.KShots(base, self.shots + self.test_shots),
                l2l.data.transforms.LoadData(base),
                l2l.data.transforms.RemapLabels(base),
                l2l.data.transforms.ConsecutiveLabels(base),
            ],
            num_tasks=self.batch_size
        )
        return meta_dataset, transform

    def _load_torchvision_dataset(self):
        dataset_cls = {
            "mnist": datasets.MNIST,
            "fashionmnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
        }.get(self.dataset_name)

        transform = self.custom_transforms or self._default_transform(self.dataset_name)
        train = self.split == "train"
        dataset = dataset_cls(self.root, train=train, transform=transform, download=self.download)
        return dataset, transform

    def _create_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_dataloader(self):
        return self.dataloader


def register_dataset(name: str, loader_fn):
    """Register a custom dataset"""
    CUSTOM_LOADERS[name.lower()] = loader_fn


# Example usage
if __name__ == "__main__":
    loader = MetaDatasetLoader(
        dataset_name="miniimagenet",
        shots=5,
        batch_size=4,
        split="train"
    )

    for task in loader.get_dataloader():
        if isinstance(task, dict):  # l2l task
            x, y = task['data'], task['labels']
        else:  # torchvision fallback
            x, y = task
        print("Data:", x.shape, "Labels:", y.shape)
        break
