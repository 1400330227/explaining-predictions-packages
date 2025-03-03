import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cem.model import ConceptEmbeddingModel
from pytorch_lightning import Trainer


def transform(img):
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # implicitly divides by 255
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    return transform(img)



def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio

class CUBDataset(Dataset):
    def __init__(self, pkl_file_paths):
        super(CUBDataset, self).__init__()
        self.data = []
        for file_path in pkl_file_paths:
            with open(file_path, 'rb') as f:
                self.data.extend(pickle.load(f))
        self.is_train = any(["train" in path for path in pkl_file_paths])
        self.root_dir = './data/'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        img_path = img_path.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/', './data/')

        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img_path = self.root_dir + '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        img = transform(img)
        attr_label = img_data['attribute_label']

        return img, class_label, torch.FloatTensor(attr_label)



train_data_path = './data/class_attr_data_10/train.pkl'
val_data_path = './data/class_attr_data_10/val.pkl'

imbalance = find_class_imbalance(train_data_path, True)
model = ConceptEmbeddingModel(n_concepts=112, embedding_activation='leakrelu', emb_size=16, n_tasks=200, c2y_model=None, weight_loss=torch.FloatTensor(imbalance),
                              task_class_weights=None,
                              concept_loss_weight=5)



train_dataset = CUBDataset(pkl_file_paths=[train_data_path])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

val_dataset = CUBDataset(pkl_file_paths=[val_data_path])
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)



trainer = Trainer(
    max_epochs=10,
    accelerator="auto",  # Auto-select GPU/CPU
    devices="auto",  # Use available hardware
)



if __name__ == '__main__':
    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
