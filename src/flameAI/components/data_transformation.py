# dataset_wrapper/preprocessors.py
from torchvision import transforms

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
