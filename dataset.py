import numpy as np
from sklearn import datasets as sklearn_datasets
import torch
from torchvision import datasets as tv_datasets
from torchvision import transforms

rng = np.random.default_rng(seed=0)

def load_dataset(dataset_name):
    """统一数据集入口"""
    if dataset_name == "mnist":
        return load_mnist_data()
    elif dataset_name == "omniglot":
        return load_omniglot_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_omniglot_data():
    """Omniglot手写字符数据集（经典小样本基准），随机选择2个类别二分类"""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten().numpy())
    ])

    train_set = tv_datasets.Omniglot(root="./data", background=True, download=True, transform=transform)
    test_set = tv_datasets.Omniglot(root="./data", background=False, download=True, transform=transform)

    all_features = []
    all_labels = []
    for i in range(len(train_set)):
        img, lbl = train_set[i]
        all_features.append(img)
        all_labels.append(lbl)
    for i in range(len(test_set)):
        img, lbl = test_set[i]
        all_features.append(img)
        all_labels.append(lbl)
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    selected_classes = rng.choice(np.unique(all_labels), size=2, replace=False)
    mask = np.isin(all_labels, selected_classes)
    features = all_features[mask]
    labels = all_labels[mask]
    labels = np.where(labels == selected_classes[0], 0, 1)

    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))
    return features, labels


def load_mnist_data():
    """MNIST手写数字数据集（sklearn的digits），仅保留0和1二分类"""
    digits = sklearn_datasets.load_digits()
    features, labels = digits.data, digits.target
    features = features[np.where((labels == 0) | (labels == 1))]
    labels = labels[np.where((labels == 0) | (labels == 1))]
    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))
    return features, labels
