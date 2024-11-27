import copy
import os.path
from functools import partial

import numpy as np
import sklearn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage.segmentation import felzenszwalb, slic, quickshift
from sklearn.linear_model import Ridge
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_pill_transform():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
    ])
    return transform


# 计算π_x (z)的值
def kernel(d, kernel_width=.25):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


# 计算欧式距离
def distances(data, x):
    return sklearn.metrics.pairwise.pairwise_distances(data, x, metric='cosine').ravel()


def get_segments(image):
    return quickshift(image, kernel_size=4, max_dist=20, ratio=0.2)


def get_data_labels(image, segments, num_samples, classifier_fn):
    fudged_image = image.copy()
    fudged_image[:] = 101
    n_features = np.unique(segments).shape[0]  # 获取索引
    data = np.array(torch.randint(0, 2, [num_samples, n_features]))
    data[0] = 1
    rows = tqdm(data)
    imgs = []
    labels = []
    batch_size = 10
    for row in rows:
        temp = copy.deepcopy(image)
        zeros = np.where(row == 0)[0]
        mask = np.zeros(segments.shape).astype(bool)

        for zero in zeros:
            mask[segments == zero] = True
        temp[mask] = fudged_image[mask]
        imgs.append(temp)
        if len(imgs) == batch_size:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
            imgs = []
    if len(imgs) > 0:
        preds = classifier_fn(np.array(imgs))
        labels.extend(preds)
    return data, np.array(labels)


def batch_predict(images, model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)

    return probs.detach().cpu().numpy()


def ridge_fit(data, labels, weights):
    clf = Ridge(alpha=0.01, fit_intercept=True, random_state=1000)
    clf.fit(data, labels, sample_weight=weights)
    return clf


def get_used_features(data, labels_column, weights):
    clf = ridge_fit(data, labels_column, weights)
    coef = clf.coef_
    weighted_data = coef * data[0]
    feature_weights = sorted(zip(range(data.shape[1]), weighted_data), key=lambda x: np.abs(x[1]), reverse=True)
    used_features = np.array([x[0] for x in feature_weights])
    return used_features


def get_explain_instance_with_data(data, labels_column, used_features, weights, top):
    easy_model = Ridge(alpha=1, fit_intercept=True, random_state=1000)
    easy_model.fit(data[:, used_features], labels_column, sample_weight=weights)
    prediction_score = easy_model.score(data[:, used_features], labels_column, sample_weight=weights)
    local_pred = easy_model.predict(data[0, used_features].reshape(1, -1))
    return (
        easy_model.intercept_, sorted(zip(used_features, easy_model.coef_), key=lambda x: np.abs(x[1]), reverse=True),
        prediction_score, local_pred)


def get_image_and_mask(label, segments, image, local_exp, num_features):
    mask = np.zeros(segments.shape).astype(segments.dtype)
    temp = copy.deepcopy(image)
    exp = local_exp[label]
    min_weight = 0.
    for f, w in exp[:num_features]:
        if np.abs(w) < min_weight:
            continue
        c = 0 if w < 0 else 1
        mask[segments == f] = -1 if w < 0 else 1
        temp[segments == f] = image[segments == f].copy()
        temp[segments == f, c] = np.max(image)
    return temp, mask


path = 'pic1.jpg'
img = get_image(path)
pill_transf = get_pill_transform()
image = np.array(pill_transf(img))

model = models.vgg16(pretrained=True)
classifier_fn = partial(batch_predict, model=model)

if __name__ == '__main__':
    segments = get_segments(image)
    num_samples = 2000
    top_labels = 5

    data, labels = get_data_labels(image, segments, num_samples, classifier_fn)
    top = np.argsort(labels[0])[-top_labels:]
    distances = distances(data, data[0].reshape(1, -1))
    num_features = data.shape[1] if data.shape[1] is not None else 30
    # num_features = 30

    ret_exp = {}
    ret_exp['intercept'] = {}
    ret_exp['local_exp'] = {}
    ret_exp['score'] = {}
    ret_exp['local_pred'] = {}
    ret_exp['top_labels'] = list(top)
    ret_exp['top_labels'].reverse()

    print(ret_exp['top_labels'])

    for label in top:
        labels_column = labels[:, label]
        used_features = get_used_features(data, labels_column, distances)
        (intercept, local_exp, score, local_pred) = get_explain_instance_with_data(data, labels_column, used_features,
                                                                                   distances, top)
        ret_exp['intercept'][label] = intercept
        ret_exp['local_exp'][label] = local_exp
        ret_exp['score'][label] = score
        ret_exp['local_pred'][label] = local_pred

    local_exp = ret_exp['local_exp']
    label = ret_exp['top_labels'][0]
    temp, mask = get_image_and_mask(label, segments, image, local_exp, num_features)

    plt.imshow(temp)
    plt.show()
