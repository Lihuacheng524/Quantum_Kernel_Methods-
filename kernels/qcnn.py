from pyvqnet.dtype import *
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.nn import Module, Parameter
from pyvqnet.qnn.vqc import QMachine
from pyvqnet.qnn.vqc.utils import probs
from pyvqnet.qnn.vqc.qcircuit import isingxx,isingyy,isingzz,u3,cnot,VQC_AmplitudeEmbedding,rxx,ryy,rzz,rzx
from pyvqnet.optim import Adam


num_wires = 6

def convolutional_layer(qm, weights, wires, skip_first_layer=True):
    n_wires = len(wires)
    assert n_wires >= 3, "this circuit is too small!"
    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                if indx % 2 == 0 and not skip_first_layer:
                    u3(q_machine=qm, wires=w, params=weights[:3])
                    u3(q_machine=qm, wires=wires[indx + 1], params=weights[3:6])
                isingxx(q_machine=qm, wires=[w, wires[indx + 1]], params=weights[6])
                isingyy(q_machine=qm, wires=[w, wires[indx + 1]], params=weights[7])
                isingzz(q_machine=qm, wires=[w, wires[indx + 1]], params=weights[8])
                u3(q_machine=qm, wires=w, params=weights[9:12])
                u3(q_machine=qm, wires=wires[indx + 1], params=weights[12:])
    return qm

def pooling_layer(qm, weights, wires):
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"
    for indx, w in enumerate(wires):
        if indx % 2 == 1 and indx < n_wires:
            cnot(q_machine=qm, wires=[w, wires[indx - 1]])
            u3(q_machine=qm, params=weights, wires=wires[indx - 1])

def conv_and_pooling(qm, kernel_weights, n_wires, skip_first_layer=True):
    convolutional_layer(qm, kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
    pooling_layer(qm, kernel_weights[15:], n_wires)
    return qm

def dense_layer(qm, weights, wires):
    rzz(q_machine=qm, params=weights[0], wires=wires)
    rxx(q_machine=qm, params=weights[1], wires=wires)
    ryy(q_machine=qm, params=weights[2], wires=wires)
    rzx(q_machine=qm, params=weights[3], wires=wires)
    rxx(q_machine=qm, params=weights[5], wires=wires)
    rzx(q_machine=qm, params=weights[6], wires=wires)
    rzz(q_machine=qm, params=weights[7], wires=wires)
    ryy(q_machine=qm, params=weights[8], wires=wires)
    rzz(q_machine=qm, params=weights[9], wires=wires)
    rxx(q_machine=qm, params=weights[10], wires=wires)
    rzx(q_machine=qm, params=weights[11], wires=wires)
    rzx(q_machine=qm, params=weights[12], wires=wires)
    rzz(q_machine=qm, params=weights[13], wires=wires)
    ryy(q_machine=qm, params=weights[14], wires=wires)
    return qm

def conv_net(qm, weights, last_layer_weights, features):
    layers = weights.shape[1]
    wires = list(range(num_wires))
    VQC_AmplitudeEmbedding(input_feature=features, q_machine=qm)
    for j in range(layers):
        conv_and_pooling(qm, weights[:, j], wires, skip_first_layer=(not j == 0))
        wires = wires[::2]
    assert last_layer_weights.size == 4 ** (len(wires)) - 1
    dense_layer(qm, last_layer_weights, wires)
    return probs(q_state=qm.states, num_wires=qm.num_wires, wires=[0])

class Qcnn_ising(Module):
    def __init__(self):
        super(Qcnn_ising, self).__init__()
        self.conv = conv_net
        self.qm = QMachine(num_wires, dtype=kcomplex128)
        self.weights = Parameter((18, 2), dtype=kfloat64)
        self.weights_last = Parameter((4 ** 2 - 1, 1), dtype=kfloat64)

    def forward(self, input):
        self.qm.reset_states(input.shape[0])
        return self.conv(self.qm, self.weights, self.weights_last, input)


def train_test_qcnn(x_train_np, y_train_np, x_test_np, y_test_np, n_epochs=10):
    model = Qcnn_ising()
    opti = Adam(model.parameters(), lr=0.01)
    x_train_tensor, x_test_tensor = QTensor(x_train_np), QTensor(x_test_np)
    for _ in range(n_epochs):
        model.train()
        opti.zero_grad()
        result = model(x_train_tensor)
        train_cost = 1.0 - tensor.sums(result[tensor.arange(0, len(y_train_np)), y_train_np]) / len(y_train_np)
        train_cost.backward()
        opti.step()
    result = model(x_train_tensor)
    train_acc = tensor.sums(result[tensor.arange(0, len(y_train_np)), y_train_np] > 0.5) / result.shape[0]
    test_out = model(x_test_tensor)
    test_acc = tensor.sums(test_out[tensor.arange(0, len(y_test_np)), y_test_np] > 0.5) / test_out.shape[0]
    return train_acc.to_numpy(), test_acc.to_numpy()
