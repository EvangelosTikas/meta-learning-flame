import os
import json
import torch
import torchvision
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram
from urllib.request import urlretrieve
from PIL import Image
import pandas as pd
import librosa
import numpy as np
from io import BytesIO
from pathlib import Path
import requests

class BaseDatasetWrapper(Dataset):
    def __init__(self, data_source, data_type=None, transform=None, cache_dir="./cache", auto_download=False):
        """
        Parameters:
        - data_source: path, URL, or torch dataset string
        - data_type: 'text', 'image', 'audio' or None (auto-detect)
        - transform: optional preprocessing transform
        - cache_dir: local cache for online data
        - auto_download: if True, will try to download data
        """
        self.data_source = data_source
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.auto_download = auto_download
        self.data_type = data_type or self._detect_type()

        self.handler = self._select_handler()

    def _detect_type(self):
        if isinstance(self.data_source, str):
            if any(self.data_source.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
                return 'image'
            elif any(self.data_source.lower().endswith(ext) for ext in ['.wav', '.mp3']):
                return 'audio'
            elif self.data_source.lower().endswith('.txt') or self.data_source.lower().endswith('.json'):
                return 'text'
            elif self.data_source in torchvision.datasets.__dict__:
                return 'image'
            elif self.data_source in torchaudio.datasets.__dict__:
                return 'audio'
        return 'unknown'

    def _select_handler(self):
        if self.data_type == 'text':
            return TextDatasetHandler(self.data_source, self.transform, self.auto_download, self.cache_dir)
        elif self.data_type == 'image':
            return ImageDatasetHandler(self.data_source, self.transform, self.auto_download, self.cache_dir)
        elif self.data_type == 'audio':
            return AudioDatasetHandler(self.data_source, self.transform, self.auto_download, self.cache_dir)
        else:
            raise ValueError(f"Unsupported or unknown data type: {self.data_type}")

    def __getitem__(self, idx):
        return self.handler[idx]

    def __len__(self):
        return len(self.handler)

class TextDatasetHandler:
    def __init__(self, source, transform=None, auto_download=False, cache_dir=Path('./cache')):
        self.transform = transform
        self.data = []

        if isinstance(source, str) and source.startswith('http') and auto_download:
            content = requests.get(source).text
            self.data = content.strip().splitlines()
        elif os.path.isfile(source):
            with open(source, 'r', encoding='utf-8') as f:
                self.data = f.readlines()
        elif isinstance(source, list):
            self.data = source
        else:
            raise ValueError("Unsupported text source")

    def __getitem__(self, idx):
        text = self.data[idx]
        return self.transform(text) if self.transform else text

    def __len__(self):
        return len(self.data)


class ImageDatasetHandler:
    def __init__(self, source, transform=None, auto_download=False, cache_dir=Path('./cache')):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if isinstance(source, str) and source in torchvision.datasets.__dict__:
            dataset_cls = torchvision.datasets.__dict__[source]
            self.dataset = dataset_cls(root=cache_dir, download=True, transform=self.transform)
        elif os.path.isdir(source):
            self.dataset = torchvision.datasets.ImageFolder(source, transform=self.transform)
        else:
            raise ValueError("Invalid image dataset source")

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class AudioDatasetHandler:
    def __init__(self, source, transform=None, auto_download=False, cache_dir=Path('./cache')):
        self.transform = transform or MelSpectrogram()

        if isinstance(source, str) and source in torchaudio.datasets.__dict__:
            dataset_cls = torchaudio.datasets.__dict__[source]
            self.dataset = dataset_cls(root=cache_dir, download=True)
        elif os.path.isdir(source):
            self.files = list(Path(source).rglob("*.wav"))
        else:
            raise ValueError("Unsupported audio dataset source")

    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):
            waveform, sample_rate, *_ = self.dataset[idx]
        else:
            waveform, sample_rate = librosa.load(self.files[idx], sr=None)
            waveform = torch.tensor(waveform).unsqueeze(0)
        return self.transform(waveform) if self.transform else waveform

    def __len__(self):
        return len(self.dataset) if hasattr(self, 'dataset') else len(self.files)


if __name__ == "__main__":
    #  example using CIFAR-10
    dataset = BaseDatasetWrapper(data_source='CIFAR10', data_type='image')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
