import json
import os

import numpy as np
import torch
from PIL import Image
from lime import lime_image
from torchvision.transforms import transforms
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LineImageExplanation(object):
    def __init__(self, model, images=None):
        super().__init__()
        self.model = model
        self.images = images
        self.explanation = None

    def get_image(self, path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as image:
                return image.convert('RGB')

    def get_input_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        return transform

    def get_pil_transform(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])

        return transform

    def get_preprocess_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return transform

    def get_input_tensor(self, img):
        transform = self.get_input_transform()
        return transform(img).unsqueeze(0)

    def batch_predict(self, images):
        self.images = images
        pil_images = [self.get_image(image) for image in images]

        get_pil_transform = self.get_pil_transform()
        pil_images_transform = [get_pil_transform(pil_image) for pil_image in pil_images]

        preprocess_transform = self.get_preprocess_transform()
        batch = torch.stack(tuple(preprocess_transform(i) for i in pil_images_transform), dim=0)

        # send model and features to device
        model = self.model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)

        return probs

    def get_image_explanation(self, fn=None):
        explainer = lime_image.LimeImageExplainer()
        get_pil_transform = self.get_pil_transform()
        explanation = explainer.explain_instance(
            np.array(get_pil_transform(self.get_image('pic1.jpg'))),
            self.batch_predict,  # classification function
            top_labels=5,
            hide_color=0,
            num_samples=1000)  # number of images that will be sent to classification function

        self.explanation = explanation
        return explanation

    def get_temp_mask(self):
        temp, mask = self.explanation.get_image_and_mask(240, positive_only=True, num_features=5, hide_rest=True)
        return temp, mask
