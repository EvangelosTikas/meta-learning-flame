# dataset_wrapper/base_dataset.py
import os
import io
import torch
import requests
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from data_transformation import get_preprocessor
from data_loaders import download_if_url
from torchvision import datasets


# TODO: add more formats
TEXT_EXT = {'.txt', '.json'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
AUDIO_EXT = {'.wav', '.mp3', '.flac'}


class MultiModalDataset(Dataset):
    def __init__(self, data_entries, task='auto', transform=None, tokenizer_name="bert-base-uncased"):
        """
        data_entries: list of dicts: {"type": "image/text/audio", "source": path_or_url, "label": ...}
        task: "image", "text", "audio", or "auto" to infer from extension
        """
        self.data_entries = data_entries
        self.task = task
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if task == "text" or task == "auto" else None

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        data_type = entry.get("type", "auto")
        source = entry["source"]
        label = entry.get("label", None)

        # Auto-detect data type
        if data_type == "auto":
            ext = os.path.splitext(source)[-1].lower()
            if ext in IMAGE_EXT:
                data_type = "image"
            elif ext in TEXT_EXT:
                data_type = "text"
            elif ext in AUDIO_EXT:
                data_type = "audio"
            else:
                raise ValueError(f"Unknown file type for {source}")

        path = download_if_url(source)

        # Load and preprocess
        if data_type == "image":
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            return img, label

        elif data_type == "text":
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            return tokens, label

        elif data_type == "audio":
            local_path = download_if_url(source)  # handles both URL and local
            waveform, sr = torchaudio.load(local_path)
            return waveform, label

        else:
            raise ValueError("Unsupported data type")


def load_builtin(name, root="./data", train=True, transform=None):
    if name == "mnist":
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    # Add more : TODO



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Sample WAV from an online source (short beep tone)
    audio_url = "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav"
    txt_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    data = [
        {"type": "image", "source": "https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png", "label": 0},
        {"type": "text", "source": txt_url, "label": 1},
        {"type": "audio", "source": audio_url, "label": "star-wars"},
    ]

    dataset = MultiModalDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    try:
        for batch in loader:
            print(batch)
    except SystemError:
        print(f"[Error]: File(s) {data} not found or corrupted.")
    except Exception as e:
        print(f"[Error]: Uknown error {e}")
        