import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import simpleshap as shap

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

model = torchvision.models.mobilenet_v2(pretrained=True, progress=False).eval().to(device)

with open('./data/imagenet_class_index.json') as file:
    class_names = [v[1] for v in json.load(file).values()]

img_path = 'pic1.jpg'

img_pil = Image.open(img_path)
X = torch.Tensor(np.array(img_pil)).unsqueeze(0)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


transform = [
    transforms.Lambda(nhwc_to_nchw),
    transforms.Resize(224),
    transforms.Lambda(lambda x: x * (1 / 255)),
    transforms.Normalize(mean=mean, std=std),
    transforms.Lambda(nchw_to_nhwc),
]

inv_transform = [
    transforms.Lambda(nhwc_to_nchw),
    transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist()
    ),
    transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)


# def predict(img: np.ndarray) -> torch.Tensor:
#     img = nhwc_to_nchw(torch.Tensor(img)).to(device)
#     output = model(img)
#     return output


def predict(img):
    img = nhwc_to_nchw(torch.Tensor(img)).to(device)
    output = model(img)
    return output


Xtr = transform(X)
out = predict(Xtr[0:1])

classes = torch.argmax(out, axis=1).detach().cpu().numpy()
print(f'Classes: {classes}: {np.array(class_names)[classes]}')

## 设置shap可解释性分析算法
input_img = Xtr[0].unsqueeze(0)

batch_size = 100

n_evals = 200  # 迭代次数越大，显著性分析粒度越精细，计算消耗时间越长

# 定义 mask，遮盖输入图像上的局部区域
masker_blur = shap.maskers.Image("blur(64, 64)", Xtr[0].shape)

# 创建可解释分析算法

if __name__ == '__main__':
    explainer = shap.Explainer(predict, masker_blur, output_names=class_names)
    shap_values = explainer(input_img, max_evals=n_evals, batch_size=batch_size, outputs=[243])

    shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]  # 原图
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]  # shap值热力图

    shap.image_plot(shap_values=shap_values.values,
                    pixel_values=shap_values.data,
                    labels=shap_values.output_names)
