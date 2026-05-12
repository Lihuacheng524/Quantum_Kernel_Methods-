import numpy as np
from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.qnn.vqc.qcircuit import hadamard, crz, rz
from pyvqnet.qnn.vqc import QMachine
from pyvqnet import tensor


def vqnet_quantum_kernel(X_1, X_2=None):
    if X_2 is None:
        X_2 = X_1
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]

    projector = np.zeros((2**N, 2**N))
    projector[0, 0] = 1
    projector = QTensor(projector, dtype=kcomplex128)

    def kernel(x1, x2):
        qm = QMachine(N, dtype=kcomplex128)
        for i in range(N):
            hadamard(q_machine=qm, wires=i)
            rz(q_machine=qm, params=QTensor(2 * x1[i], dtype=kfloat64), wires=i)
        for i in range(N):
            for j in range(i + 1, N):
                crz(q_machine=qm, params=QTensor(2 * (np.pi - x1[i]) * (np.pi - x1[j]), dtype=kfloat64), wires=[i, j])
        for i in range(N):
            for j in range(i + 1, N):
                crz(q_machine=qm, params=QTensor(2 * (np.pi - x2[i]) * (np.pi - x2[j]), dtype=kfloat64), wires=[i, j], use_dagger=True)
        for i in range(N):
            rz(q_machine=qm, params=QTensor(2 * x2[i], dtype=kfloat64), wires=i, use_dagger=True)
            hadamard(q_machine=qm, wires=i, use_dagger=True)

        states_1 = qm.states.reshape((1, -1))
        states_1 = tensor.conj(states_1)
        states_2 = qm.states.reshape((-1, 1))
        result = tensor.matmul(tensor.conj(states_1), projector)
        result = tensor.matmul(result, states_2)
        return result.to_numpy()[0][0].real

    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(len(X_1)):
        for j in range(len(X_2)):
            gram[i][j] = kernel(X_1[i], X_2[j])
    return gram
