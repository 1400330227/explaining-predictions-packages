import os

import numpy as np
import torch
from PIL import Image
from lime import lime_image
from torchvision.transforms import transforms
from ultralytics import YOLO

import torch.nn.functional as F

model = YOLO("yolo11n.pt")
results = model('test.jpg', save=True)
results[0].show()


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

path = ''
img = get_image(path)
explainer = lime_image.LimeImageExplainer()


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)

    return probs.detach().cpu().numpy()


explanation = explainer.explain_instance(np.array(pill_transf(img)), batch_predict, top_labels=5, hide_color=0,
                                         num_samples=1000)
