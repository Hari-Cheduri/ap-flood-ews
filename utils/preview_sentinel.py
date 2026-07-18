import numpy as np
from PIL import Image
import os

arr = np.load("data/processed/sentinel1_ap_patch.npy")

img = (arr * 255).clip(0, 255).astype("uint8")

os.makedirs("outputs", exist_ok=True)

Image.fromarray(img).save("outputs/sentinel1_preview.png")

print("Saved preview: outputs/sentinel1_preview.png")
print("Shape:", arr.shape)
print("Min:", arr.min())
print("Max:", arr.max())
print("Mean:", arr.mean())