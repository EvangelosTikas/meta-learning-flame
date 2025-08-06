# dataset_wrapper/preprocessors.py
from torchvision import transforms
import learn2learn

CACHE = {}


# ------------
# Standard Transforms
# (Training)

TRAIN_AUGMENT_1 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomCrop(128, padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(127.5,127.5)
])

TRAIN_IMG_1 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.Normalize(127.5,127.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# - omniglot_transforms.py
omniglot_transforms = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.92206,), (0.08426,))  # mean/std from dataset
])
# - miniimagenet_transforms.py
miniimagenet_transforms = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.472, 0.451, 0.403], std=[0.278, 0.268, 0.282])
])
# - cifarfs_transforms.py
cifarfs_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                         std=[0.2673, 0.2564, 0.2762])
])
# - quickdraw_transforms.py
quickdraw_transforms = transforms.Compose([
    transforms.Resize(28),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# ------------
# Meta-learning data Transforms
# (Meta)


base = learn2learn.data.MetaDataset()

META_TASK_1 =[
                learn2learn.data.transforms.NWays(base, 5),
                learn2learn.data.transforms.KShots(base, 1 + 15),
                learn2learn.data.transforms.LoadData(base),
                learn2learn.data.transforms.RemapLabels(base),
                learn2learn.data.transforms.ConsecutiveLabels(base),
            ]


META_FC100_1 = train_transforms = [
                learn2learn.data.transforms.FusedNWaysKShots(base, n=ways, k=2*shots),
                learn2learn.data.transforms.LoadData(base),
                learn2learn.data.transforms.RemapLabels(base),
                learn2learn.data.transforms.ConsecutiveLabels(base),
            ]


# === Meta Task Transforms ===

def get_task_transforms(dataset, n_way=5, k_shot=1, k_query=15, shuffle=True):
    """
    Returns a list of learn2learn TaskTransforms for episodic meta-learning.
    """
    trnsf_arr : learn2learn.data.transforms.TaskTransform = []
    return [
        learn2learn.data.transforms.NWays(dataset, n=n_way),
        learn2learn.data.transforms.KShots(dataset, k=k_shot + k_query),
        learn2learn.data.transforms.LoadData(dataset),
        learn2learn.data.transforms.RemapLabels(dataset),
        learn2learn.data.transforms.FusedNWaysKShots(dataset, n=n_way, k=k_shot + k_query, shuffle=shuffle),
    ]



def get_preprocessor(data_type):
    if data_type == "image":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif data_type == "text":
        # text handled by tokenizer
        return None
    elif data_type == "audio":
        return None  # torchaudio handles this
    else:
        return None


def get_custom_transform(name : str = "base", tag = [0,0], transforms = []):
    # Under construction : TODO

    return transforms.Compose([
            transforms,
        ])
    return 0



if __name__ == "__main__":
    get_prp = get_preprocessor("image")
