import pickle
import time

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from template_models import InceptionV3, MLP, End2EndModel
import rpds

random.seed(1)
np.random.seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # implicitly divides by 255
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
])


def get_pkl_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        return pickle.load(f)


class Datasets(Dataset):
    def __init__(self, pkl_file_paths):
        super(Datasets, self).__init__()
        self.data = []
        for path in pkl_file_paths:
            data = get_pkl_data(path)
            self.data.extend(data)
        # self.data = self.data[0:200]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        attribute_label = img_data['attribute_label']
        class_label = img_data['class_label']
        try:
            index = img_path.split('/').index('CUB_200_2011')
            img_path = '/'.join(img_path.split('/')[index:])
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            return img, class_label, attribute_label, img_path
        except Exception as e:
            print(e)


datasets = Datasets(['./CUB_processed/class_attr_data_10/train.pkl', './CUB_processed/class_attr_data_10/val.pkl'])
train_loader = DataLoader(datasets, batch_size=16, shuffle=True)


def ModelXtoCtoY(n_attributes, num_classes, use_aux, expand_dim, use_relu, use_sigmoid, n_class_attr):
    model1 = InceptionV3(num_classes, n_attributes, expand_dim, use_aux)
    model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)


def find_class_imbalance(train_data_path):
    imbalance_ratio = []
    data = get_pkl_data(train_data_path)
    n = len(data)
    n_attr = len(data[0]['attribute_label'])

    n_ones = [0] * n_attr
    total = [n] * n_attr

    for d in data:
        labels = d['attribute_label']
        for i in range(n_attr):
            n_ones[i] += labels[i]

    for j in range(n_attr):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)

    return imbalance_ratio


attr_loss_weight = 0.01
n_attributes = 112
num_epoch = 10
num_classes = 200
expand_dim = 0
n_class_attr = 2
use_aux = True

imbalance = find_class_imbalance('./CUB_processed/class_attr_data_10/train.pkl')
model = ModelXtoCtoY(n_attributes, num_classes, use_aux, expand_dim, False, False, 2)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
attr_criterion = []

for ratio in imbalance:
    attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.tensor([ratio]).to(device)))

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9,
                            weight_decay=0.0004)
model.train()

for epoch in range(num_epoch):
    train_losses, valid_losses, train_acc, valid_acc, = [], [], [], []
    start_time = time.time()
    for _, data, in enumerate(train_loader):
        inputs, labels, attr_labels, img_path = data
        attr_labels = [i.long() for i in attr_labels]
        attr_labels = torch.stack(attr_labels).t()  # .float() #N x 312

        attr_labels_var = torch.autograd.Variable(attr_labels).float()
        attr_labels_var = attr_labels_var.to(device)

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.to(device)

        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.to(device)

        outputs, aux_outputs = model(inputs_var)
        losses = []

        loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
        losses.append(loss_main)

        train_accuracy = torch.eq(torch.argmax(outputs[0].squeeze(), dim=1), labels_var).float().mean()
        train_acc.append(train_accuracy.item())

        for i in range(len(attr_criterion)):
            losses.append(attr_loss_weight * (
                    1.0 * attr_criterion[i](outputs[i + 1].squeeze().type(torch.cuda.FloatTensor),
                                            attr_labels_var[:, i]) +
                    0.4 * attr_criterion[i](aux_outputs[i + 1].squeeze().type(torch.cuda.FloatTensor),
                                            attr_labels_var[:, i])))

        total_loss = losses[0] + sum(losses[1:])
        total_loss = total_loss / (1 + attr_loss_weight * n_attributes)
        train_losses.append(total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    duration = time.time() - start_time
    print('Epoch[{}/{}], Duration:{:.8f}, Loss:{:.8f}, Train_Accuracy:{:.5f}'.format(
        epoch + 1, num_epoch, duration, np.mean(train_losses), np.mean(train_acc)))
torch.save(model, 'best_model.pth')