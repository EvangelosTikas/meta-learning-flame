from torchvision.datasets import ImageFolder
import learn2learn
from learn2learn.data import MetaDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision import transforms

# Suppose Custom dataset folder: data/custom/{class}/images...
dataset = ImageFolder('data/custom', transform=transforms.ToTensor())

class MetaDatasetCustom:
    def __init__(self, dataset_name,
                 root='data',
                 split='train',
                 shuffle=True,
                 download=True,
                 custom_transforms=None):

        self.dataset_name = dataset_name.lower()
        self.root = root
        self.shuffle = shuffle
        self.download = download
        self.custom_transforms = custom_transforms

        self.dataset = self._resolve_dataset()


    def get(self):
        return None


    def set(self):
        return None


# Example usage
if __name__ == "__main__":
    # Suppose Custom dataset folder: data/custom/{class}/images...
    dataset = ImageFolder('data/custom', transform=transforms.ToTensor())

    meta = MetaDataset(dataset)
    transform = transforms.Compose([
        NWays(meta, n=5),
        KShots(meta, k=1 + 15),   # 1 support + K query
        LoadData(meta),
        RemapLabels(meta),
    ])
    taskset = learn2learn.data.TaskDataset(meta, task_transforms=[transform], num_tasks=1000)
    support, query = taskset.sample()
