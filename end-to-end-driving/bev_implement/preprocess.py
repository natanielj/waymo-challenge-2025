import torch
import numpy as np
from PIL import Image
import io
import torchvision.transforms as T


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img)