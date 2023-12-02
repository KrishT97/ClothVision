import os
import time

import numpy as np
import torch
from PIL import Image

label_test = Image.open(os.path.join("./data/labels/pixel_level_labels_colored/0001.png"))

array_to_onehot_random = {}  # random dic to simulate how cost it would be
lista = []
for color in np.unique(np.asarray(label_test, dtype=np.int32).reshape(-1, 3), axis=0):
    array_to_onehot_random[color.tobytes()] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def preprocess(img):
    img = np.asarray(img, dtype=np.int32)
    codified = np.apply_along_axis(lambda pixel: array_to_onehot_random[pixel.tobytes()], 2, img)
    return torch.tensor(codified)


start = time.time()
preprocess(label_test)
final = time.time() - start
print("Tardo", final, "segundos para una imagen.")
print("Para las 1004 imágenes, tardaría", final * 1004, "cada una de las épocas sólo en preprocesado.")
