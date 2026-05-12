import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy.linalg as la


import sys
sys.path.insert(0, "./")
import pyvqnet
from pyvqnet import _core
from pyvqnet.dtype import *

from pyvqnet.tensor.tensor import QTensor
from pyvqnet.qnn.vqc.qcircuit import PauliZ, VQC_ZZFeatureMap,PauliX,PauliY,hadamard,crz,rz
from pyvqnet.qnn.vqc import QMachine
from pyvqnet.qnn.vqc.qmeasure import MeasureAll
from pyvqnet import tensor
import functools as ft

np.random.seed(42)
# data load
digits = datasets.load_digits(n_class=2)
# create lists to save the results
gaussian_accuracy = []
quantum_accuracy = []
projected_accuracy = []
quantum_gaussian = []
projected_gaussian = []

# reduce dimensionality

def custom_data_map_func(x):
    """
    custom data map function
    """
    coeff = x[0] if x.shape[0] == 1 else ft.reduce(lambda m, n: m * n, x)
    return coeff.reshape([1])

def vqnet_quantum_kernel(X_1, X_2=None):

    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]

    # create projector (measures probability of having all "00...0")
    projector = np.zeros((2**N, 2**N))
    projector[0, 0] = 1
    projector = QTensor(projector,dtype=kcomplex128)
    # define the circuit for the quantum kernel ("overlap test" circuit)

    def kernel(x1, x2):
        qm = QMachine(N, dtype=kcomplex128)

        for i in range(N):
            hadamard(q_machine=qm, wires=i)
            rz(q_machine=qm,params=QTensor(2 * x1[i],dtype=kfloat64), wires=i)
        for i in range(N):
            for j in range(i + 1, N):
                crz(q_machine=qm,params=QTensor(2 * (np.pi - x1[i]) * (np.pi - x1[j]),dtype=kfloat64), wires=[i, j])

        for i in range(N):
            for j in range(i + 1, N):
                crz(q_machine=qm,params=QTensor(2 * (np.pi - x2[i]) * (np.pi - x2[j]),dtype=kfloat64), wires=[i, j],use_dagger=True)
        for i in range(N):
            rz(q_machine=qm,params=QTensor(2 * x2[i],dtype=kfloat64), wires=i,use_dagger=True)
            hadamard(q_machine=qm, wires=i,use_dagger=True)

        states_1 = qm.states.reshape((1,-1))
        states_1 = tensor.conj(states_1)

        states_2 = qm.states.reshape((-1,1))

        result = tensor.matmul(tensor.conj(states_1), projector)
        result = tensor.matmul(result, states_2)
        return result.to_numpy()[0][0].real

    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(len(X_1)):
        for j in range(len(X_2)):
            gram[i][j] = kernel(X_1[i], X_2[j])

    return gram


def vqnet_projected_quantum_kernel(X_1, X_2=None, params=QTensor([1.0])):

    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"


    def projected_xyz_embedding(X):

        N = X.shape[1]

        def proj_feature_map(x):
            qm = QMachine(N, dtype=kcomplex128)
            VQC_ZZFeatureMap(x, qm, data_map_func=custom_data_map_func, entanglement="linear")

            return (
                [MeasureAll(obs={f"X{i}":1})(qm) for i in range(N)]
                + [MeasureAll(obs={f"Y{i}":1})(qm) for i in range(N)]
                + [MeasureAll(obs={f"Z{i}":1})(qm) for i in range(N)]
            )

        # build the gram matrix
        X_proj = [proj_feature_map(x) for x in X]

        return X_proj
    X_1_proj = projected_xyz_embedding(QTensor(X_1))
    X_2_proj = projected_xyz_embedding(QTensor(X_2))


    # build the gram matrix

    gamma = params[0]
    gram = tensor.zeros(shape=[X_1.shape[0], X_2.shape[0]],dtype=kfloat64)

    for i in range(len(X_1_proj)):
        for j in range(len(X_2_proj)):
            result = [a - b for a,b in zip(X_1_proj[i], X_2_proj[j])]
            result = [a**2 for a in result]
            value = tensor.exp(-gamma * sum(result).squeeze(0))
            gram[i,j] = value
    return gram


def calculate_generalization_accuracy(
    training_gram, training_labels, testing_gram, testing_labels
):

    svm = SVC(kernel="precomputed")
    svm.fit(training_gram, training_labels)

    y_predict = svm.predict(testing_gram)
    correct = np.sum(testing_labels == y_predict)
    accuracy = correct / len(testing_labels)
    return accuracy

import time
qubits = [2, 4, 8]

for n in qubits:
    n_qubits = n
    x_tr, x_te , y_tr , y_te = train_test_split(digits.data, digits.target, test_size=0.3, random_state=22)

    pca = PCA(n_components=n_qubits).fit(x_tr)
    x_tr_reduced = pca.transform(x_tr)
    x_te_reduced = pca.transform(x_te)

    # normalize and scale

    std = StandardScaler().fit(x_tr_reduced)
    x_tr_norm = std.transform(x_tr_reduced)
    x_te_norm = std.transform(x_te_reduced)

    samples = np.append(x_tr_norm, x_te_norm, axis=0)
    minmax = MinMaxScaler((-1,1)).fit(samples)
    x_tr_norm = minmax.transform(x_tr_norm)
    x_te_norm = minmax.transform(x_te_norm)

    # select only 100 training and 20 test data

    tr_size = 100
    x_tr = x_tr_norm[:tr_size]
    y_tr = y_tr[:tr_size]

    te_size = 20
    x_te = x_te_norm[:te_size]
    y_te = y_te[:te_size]

    quantum_kernel_tr = vqnet_quantum_kernel(X_1=x_tr)

    projected_kernel_tr = vqnet_projected_quantum_kernel(X_1=x_tr)

    quantum_kernel_te = vqnet_quantum_kernel(X_1=x_te, X_2=x_tr)

    projected_kernel_te = vqnet_projected_quantum_kernel(X_1=x_te, X_2=x_tr)

    quantum_accuracy.append(calculate_generalization_accuracy(quantum_kernel_tr, y_tr, quantum_kernel_te, y_te))
    print(f"qubits {n}, quantum_accuracy {quantum_accuracy[-1]}")
    projected_accuracy.append(calculate_generalization_accuracy(projected_kernel_tr.to_numpy(), y_tr, projected_kernel_te.to_numpy(), y_te))
    print(f"qubits {n}, projected_accuracy {projected_accuracy[-1]}")

# train_size 100 test_size 20
#
# qubits 2, quantum_accuracy 1.0
# qubits 2, projected_accuracy 1.0
# qubits 4, quantum_accuracy 1.0
# qubits 4, projected_accuracy 1.0
# qubits 8, quantum_accuracy 0.45
# qubits 8, projected_accuracy 1.0

# train_size 100 test_size 100
#
# qubits 2, quantum_accuracy 1.0
# qubits 2, projected_accuracy 0.99
# qubits 4, quantum_accuracy 0.99
# qubits 4, projected_accuracy 0.98
# qubits 8, quantum_accuracy 0.51
# qubits 8, projected_accuracy 0.99