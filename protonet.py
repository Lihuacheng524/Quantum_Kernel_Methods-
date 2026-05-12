import numpy as np
from sklearn.metrics import accuracy_score


def train_test_protonet(x_train, y_train, x_test, y_test):
    """原型网络：使用训练集作为支持集，计算类别原型并预测"""
    classes = np.unique(y_train)
    prototypes = np.array([x_train[y_train == c].mean(axis=0) for c in classes])
    test_dists = np.linalg.norm(x_test[:, None, :] - prototypes[None, :, :], axis=2)
    train_dists = np.linalg.norm(x_train[:, None, :] - prototypes[None, :, :], axis=2)

    y_predict = classes[test_dists.argmin(axis=1)]
    train_predict = classes[train_dists.argmin(axis=1)]

    test_acc = accuracy_score(y_test, y_predict)
    train_acc = accuracy_score(y_train, train_predict)
    return train_acc, test_acc
