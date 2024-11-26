import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision import transforms, models
import torch.nn.functional as F
# from lime import lime_image

from simplelime import lime_image


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')


path = 'pic1.jpg'
data_path = './data/imagenet_class_index.json'
img = get_image(path)


# plt.figure()
# plt.imshow(img)
# plt.show()  # display it


def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def get_input_tensor(img):
    transform = get_input_transform()
    return transform(img).unsqueeze(0)


# load the pretrained model for Resnet50 available
model = models.vgg16(pretrained=True)

idx2label, cls2label, cls2idx = [], {}, {}

with open(os.path.abspath(data_path), 'rb') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

img_t = get_input_tensor(img)
print(model.eval())

logits = model(img_t)
probs = F.softmax(logits, dim=1)
probs5 = probs.topk(5)
tuple((p, c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


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


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)

    return probs.detach().cpu().numpy()


test_pred = batch_predict([pill_transf(img)])

prob = test_pred.squeeze().argmax()
print('{} -> {}'.format(prob, class_idx[str(prob)]))

if __name__ == '__main__':
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict,  # classification function
                                             top_labels=20,
                                             hide_color=101,
                                             random_seed=1000,
                                             num_samples=10000)  # number of images that will be sent to classification function

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=30,
    #                                             hide_rest=False)
    # img_boundry1 = mark_boundaries(temp / 255.0, mask)
    # plt.imshow(img_boundry1)

    # plt.show()

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=30,
                                                hide_rest=False)

    plt.imshow(temp)
    plt.show()
    plt.imshow(mask)
    plt.show()

    img_boundry2 = mark_boundaries(temp, mask)
    plt.imshow(img_boundry2)

    plt.show()

    img_boundry3 = mark_boundaries(temp / 255.0, mask)
    plt.imshow(img_boundry3)

    plt.show()
