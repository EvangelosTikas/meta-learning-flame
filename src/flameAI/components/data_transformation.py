# dataset_wrapper/preprocessors.py
from torchvision import transforms

CACHE = {}

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
