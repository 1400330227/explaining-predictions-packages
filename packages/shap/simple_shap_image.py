import os.path

import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic
from torchvision.transforms import transforms


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

path = 'pic1.jpg'
image = get_image(path)

def get_pil_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
    ])

    return transform

pil_transform = get_pil_transform()
pil_image = pil_transform(image)

segments_slic = slic(pil_image, n_segments=10, compactness=30, sigma=3)

plt.imshow(segments_slic)
plt.show()