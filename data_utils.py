# (translated)  data_utils.py
# (translated)  Data generator que carrega imagens, aplica filtros e retorna batches
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from filters import apply_filters_as_channels

class FilteredImageSequence(Sequence):

    def __init__(self, root_dir, batch_size=32, size=(128,128), shuffle=True, use_filters=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle
        self.use_filters = use_filters
        self.samples = []
        self.class_indices = {}
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for i, cls in enumerate(classes):
            self.class_indices[cls] = i
            cls_dir = os.path.join(root_dir, cls)
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.jpg','.jpeg','.png','bmp')):
                    self.samples.append((os.path.join(cls_dir, fn), i))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size: (idx+1) * self.batch_size]
        X_list = []
        y = []
        for fp, label in batch:
            img = Image.open(fp).convert('RGB')
            if self.use_filters:
                arr = apply_filters_as_channels(img, size=self.size)  # (translated)  shape H,W,channels
            else:
                arr = np.array(img.resize(self.size)).astype(np.float32) / 255.0
            X_list.append(arr)
            y.append(label)
        X = np.stack(X_list, axis=0)
        y = np.array(y)
       
        from tensorflow.keras.utils import to_categorical
        return X, to_categorical(y, num_classes=len(self.class_indices))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)
