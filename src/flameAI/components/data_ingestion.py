import json
import learn2learn
import os
from torchvision.datasets import ImageFolder
from learn2learn.data import MetaDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision import transforms

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json
from typing import Union, Dict, List, Optional

class UniversalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        modalities: Dict[str, Union[str, List[str]]],
        label_column: str = "label",
        transforms: Optional[Dict[str, callable]] = None,
    ):
        """
        Args:
            data_dir (str): Directory containing all data files
            label_file (str): Path to labels file (CSV/JSON/JSONL)
            modalities (Dict): {
                "image": "image_column_name",
                "text": "text_column_name",
                "tabular": ["col1", "col2"]
            }
            label_column (str): Name of label column
            transforms (Dict): {
                "image": transform_function,
                "text": transform_function,
                "tabular": transform_function
            }
        """
        self.data_dir = data_dir
        self.modalities = modalities
        self.transforms = transforms or {}
        self.label_column = label_column

        # Auto-detect file type and load labels
        self.label_data = self._load_label_file(label_file)

        # Validate columns
        self._validate_columns()

    def _load_label_file(self, label_file: str) -> List[Dict]:
        """Auto-detect and load label file"""
        ext = os.path.splitext(label_file)[1].lower()

        if ext == '.csv':
            return pd.read_csv(label_file).to_dict('records')
        elif ext == '.json':
            with open(label_file, 'r') as f:
                return json.load(f)
        elif ext == '.jsonl':
            with open(label_file, 'r') as f:
                return [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported label file format: {ext}")

    def _validate_columns(self):
        """Verify all specified columns exist"""
        sample = self.label_data[0]
        for modality, columns in self.modalities.items():
            if isinstance(columns, str):
                if columns not in sample:
                    raise KeyError(f"Column '{columns}' for {modality} not found")
            else:
                for col in columns:
                    if col not in sample:
                        raise KeyError(f"Column '{col}' for {modality} not found")

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        item = self.label_data[idx]
        data = {}

        # Handle image data
        if 'image' in self.modalities:
            img_path = os.path.join(self.data_dir, item[self.modalities['image']])
            image = Image.open(img_path).convert('RGB')
            if 'image' in self.transforms:
                image = self.transforms['image'](image)
            data['image'] = image

        # Handle text data
        if 'text' in self.modalities:
            text = item[self.modalities['text']]
            if 'text' in self.transforms:
                text = self.transforms['text'](text)
            data['text'] = text

        # Handle tabular data
        if 'tabular' in self.modalities:
            tabular = {col: item[col] for col in self.modalities['tabular']}
            if 'tabular' in self.transforms:
                tabular = self.transforms['tabular'](tabular)
            data['tabular'] = tabular

        # Get label
        label = item[self.label_column]

        return data, label

# Example Flexible Modality Support:

# python
# # Choose any combination:
# modalities = {
#     'image': 'image_path_column',
#     'text': 'text_column',
#     'tabular': ['feature1', 'feature2']
# }
# Modality-Specific Transforms:

# python
# transforms = {
#     'image': torchvision.transforms.Compose([...]),
#     'text': lambda x: tokenizer(x, ...),
#     'tabular': normalize_tabular_data
# }



# Example usage
if __name__ == "__main__":

    modalities = {
        'image': 'image_path_column',
        'text': 'text_column',
        'tabular': ['feature1', 'feature2']
        }

    # Example

    # # 1. Image + Text
    # imtxt_dataset = UniversalDataset(
    #     data_dir='data/',
    #     label_file='labels.json',
    #     modalities={
    #         'image': 'image_path',
    #         'text': 'caption'
    #     },
    #     transforms={
    #         'image': image_transform,
    #         'text': text_tokenizer
    #     }
    # )
    # # 2. Tabular Only
    # tab_dataset = UniversalDataset(
    #     data_dir='data/',
    #     label_file='labels.csv',
    #     modalities={
    #         'tabular': ['age', 'income', 'score']
    #     }
    # )
    # # 3. All Three Modalities
    # mod3_dataset = UniversalDataset(
    #     data_dir='multimodal_data/',
    #     label_file='labels.jsonl',
    #     modalities={
    #         'image': 'img_file',
    #         'text': 'description',
    #         'tabular': ['price', 'weight', 'rating']
    #     },
    #     transforms={
    #         'image': resize_and_normalize,
    #         'text': bert_tokenizer,
    #         'tabular': scale_features
    #     }
    # )
