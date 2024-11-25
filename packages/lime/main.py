import json
import os

from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from torchvision import models

from line_image import LineImageExplanation

data_path = './data/imagenet_class_index.json'


def get_classes(data_path):
    class_idx, idx2label, cls2label, cls2idx = {}, [], {}, {}
    with open(os.path.abspath(data_path), 'rb') as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    return class_idx, idx2label, cls2label, cls2idx


def main():
    model = models.inception_v3(pretrained=True)
    model.eval()  # It means validation mode

    explainer = LineImageExplanation(model)
    probs = explainer.batch_predict(['pic1.jpg']).squeeze()
    probs5 = probs.topk(5)

    explanation = explainer.get_image_explanation()
    temp, mask = explainer.get_temp_mask()
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

    class_idx, idx2label, cls2label, cls2idx = get_classes(data_path)

    for c, p in zip(probs5.indices.cpu().data.numpy(), probs5.values.cpu().data.numpy()):
        print(tuple((p, c, idx2label[c])))

    # print(model.eval())


if __name__ == '__main__':
    main()
