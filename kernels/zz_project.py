import numpy as np
import functools as ft
from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.qnn.vqc.qcircuit import VQC_ZZFeatureMap
from pyvqnet.qnn.vqc import QMachine
from pyvqnet.qnn.vqc.qmeasure import MeasureAll
from pyvqnet import tensor


def custom_data_map_func(x):
    coeff = x[0] if x.shape[0] == 1 else ft.reduce(lambda m, n: m * n, x)
    return coeff.reshape([1])


def vqnet_projected_quantum_kernel(X_1, X_2=None, params=QTensor([1.0])):
    if X_2 is None:
        X_2 = X_1
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"

    def projected_xyz_embedding(X):
        N = X.shape[1]
        def proj_feature_map(x):
            qm = QMachine(N, dtype=kcomplex128)
            VQC_ZZFeatureMap(x, qm, data_map_func=custom_data_map_func, entanglement="linear")
            return (
                [MeasureAll(obs={f"X{i}": 1})(qm) for i in range(N)]
                + [MeasureAll(obs={f"Y{i}": 1})(qm) for i in range(N)]
                + [MeasureAll(obs={f"Z{i}": 1})(qm) for i in range(N)]
            )
        return [proj_feature_map(x) for x in X]

    X_1_proj = projected_xyz_embedding(QTensor(X_1))
    X_2_proj = projected_xyz_embedding(QTensor(X_2))

    gamma = params[0]
    gram = tensor.zeros(shape=[X_1.shape[0], X_2.shape[0]], dtype=kfloat64)
    for i in range(len(X_1_proj)):
        for j in range(len(X_2_proj)):
            result = [a - b for a, b in zip(X_1_proj[i], X_2_proj[j])]
            result = [a**2 for a in result]
            value = tensor.exp(-gamma * sum(result).squeeze(0))
            gram[i, j] = value
    return gram
