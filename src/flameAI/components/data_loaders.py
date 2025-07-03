# No changes should be made to the original code

# components/data_loaders.py
import os
import requests
import tempfile
import importlib
import warnings
# For ready-to-use datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_CACHE = {}

def download_if_url(path_or_url):
    if path_or_url.startswith("http"):
        if path_or_url in _CACHE:
            return _CACHE[path_or_url]

        response = requests.get(path_or_url)
        response.raise_for_status()

        ext = os.path.splitext(path_or_url)[-1]
        suffix = ext if ext in {'.wav', '.mp3', '.flac'} else ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()

        _CACHE[path_or_url] = tmp.name
        return tmp.name
    else:
        return path_or_url


try:
    from torchmeta.datasets import helpers as torchmeta_helpers
    from torchmeta.utils.data import BatchMetaDataLoader
    from torchmeta.datasets.helpers import miniimagenet

    HAS_TORCHMETA = True
except ImportError:
    HAS_TORCHMETA = False
    warnings.warn("torchmeta not installed. Falling back to minimal dataset support.")


class MetaDatasetLoader:
    def __init__(self, dataset_name, root='data',
                 ways=5, shots=1, test_shots=15,
                 split='train', batch_size=16, shuffle=True,
                 download=True, use_torchmeta=True):

        self.dataset_name = dataset_name.lower()
        self.root = root
        self.ways = ways
        self.shots = shots
        self.test_shots = test_shots
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.download = download
        self.use_torchmeta = use_torchmeta and HAS_TORCHMETA

        if self.use_torchmeta:
            self.dataset = self._load_torchmeta_dataset()
            self.dataloader = BatchMetaDataLoader(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.shuffle)
        else:
            self.dataset = self._load_basic_dataset()
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.batch_size,
                                         shuffle=self.shuffle)

    def _load_torchmeta_dataset(self):
        if not hasattr(torchmeta_helpers, self.dataset_name):
            raise ValueError(f"Dataset '{self.dataset_name}' not found in torchmeta.helpers.")
        dataset_fn = getattr(torchmeta_helpers, self.dataset_name)
        return dataset_fn(self.root,
                          ways=self.ways,
                          shots=self.shots,
                          test_shots=self.test_shots,
                          meta_train=self.split == 'train',
                          meta_val=self.split == 'val',
                          meta_test=self.split == 'test',
                          download=self.download)

    def _load_basic_dataset(self):
        # Simulate few-shot tasks using torchvision datasets (e.g., MNIST)
        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor()
            ])
            train = self.split == 'train'
            return datasets.MNIST(self.root, train=train, transform=transform, download=self.download)

        raise NotImplementedError(f"Basic fallback for dataset '{self.dataset_name}' not implemented.")

    def get_dataloader(self):
        return self.dataloader



def example_use_dataset(dataset_name: str = "omniglot"):
    if HAS_TORCHMETA:
        # Use torchmeta if installed
        loader = MetaDatasetLoader(dataset_name, split="val", shots=5, use_torchmeta=True)
        for task in loader.get_dataloader():
            print(task['train'][0].shape)
            break

    else:
        # Use fallback with torchvision
        mnist_loader = MetaDatasetLoader("mnist", split="train", use_torchmeta=False)
        for batch in mnist_loader.get_dataloader():
            x, y = batch
            print(x.shape, y.shape)
            break


if __name__ == "__main__":

    # Load training tasks
    dataset = miniimagenet("data",
                        ways=5,
                        shots=1,
                        test_shots=15,
                        meta_train=True,
                        download=True)

    # Create a task-level data loader
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate over tasks (each batch is a task)
    for batch in dataloader:
        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]
        print("Train shape:", train_inputs.shape)
        print("Test shape:", test_inputs.shape)
        break
