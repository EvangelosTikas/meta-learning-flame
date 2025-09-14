from learn2learn.data import MetaDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision import transforms

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json
from typing import Union, Dict, List, Optional


Train_Params = {
    "Train"   : 1,
    "Validate": 0,
    "LR"      : 10e-3,
    "Optimizer" : "Adam",
    "Meta-model": "maml",

    "Test"     : 1
                }


class Learner():
    def __init__(self, parent=True):
        # set params
        self.parent = parent

    def learn(self):
        return None

class Train():
    # Empty
    def __init__(self, parent=True):
        # set params
        self.parent = parent
