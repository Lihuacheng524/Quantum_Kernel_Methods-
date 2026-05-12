import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns

from pyqpanda3 import *
from pyvqnet.qnn.vqc.qcircuit import isingxx,isingyy,isingzz,u3,cnot,VQC_AmplitudeEmbedding,rxx,ryy,rzz,rzx
from pyvqnet.qnn.vqc.qmachine import QMachine
from pyvqnet.qnn.vqc.utils import probs
from pyvqnet.nn import Module, Parameter
from pyvqnet.tensor import tensor
from pyvqnet.tensor import QTensor
from pyvqnet.dtype import *
from pyvqnet.optim import Adam

sns.set()

seed = 0
rng = np.random.default_rng(seed=seed)
n_reps = 10
n_test = 10
n_epochs = 10

def convolutional_layer(qm, weights, wires, skip_first_layer=True):

    n_wires = len(wires)
    assert n_wires >= 3, "this circuit is too small!"
    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                if indx % 2 == 0 and not skip_first_layer:

                    u3(q_machine=qm, wires=w, params=weights[:3])
                    u3(q_machine=qm, wires=wires[indx + 1], params=weights[3:6])

                isingxx(q_machine=qm,  wires=[w, wires[indx + 1]], params=weights[6])
                isingyy(q_machine=qm,  wires=[w, wires[indx + 1]], params=weights[7])
                isingzz(q_machine=qm,  wires=[w, wires[indx + 1]], params=weights[8])
                u3(q_machine=qm, wires=w, params=weights[9:12])
                u3(q_machine=qm, wires=wires[indx + 1], params=weights[12:])

    return qm

def pooling_layer(qm, weights, wires):
    """Adds a pooling layer to a circuit."""
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"
    for indx, w in enumerate(wires):
        if indx % 2 == 1 and indx < n_wires:
            cnot(q_machine=qm, wires=[w, wires[indx - 1]])
            u3(q_machine=qm, params=weights, wires=wires[indx - 1])

def conv_and_pooling(qm, kernel_weights, n_wires, skip_first_layer=True):
    """Apply both the convolutional and pooling layer."""

    convolutional_layer(qm, kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
    pooling_layer(qm, kernel_weights[15:], n_wires)
    return qm

def dense_layer(qm, weights, wires):
    """Apply an arbitrary unitary gate to a specified set of wires."""

    rzz(q_machine=qm,params=weights[0], wires=wires)
    rxx(q_machine=qm,params=weights[1], wires=wires)
    ryy(q_machine=qm,params=weights[2], wires=wires)
    rzx(q_machine=qm,params=weights[3], wires=wires)
    rxx(q_machine=qm,params=weights[5], wires=wires)
    rzx(q_machine=qm,params=weights[6], wires=wires)
    rzz(q_machine=qm,params=weights[7], wires=wires)
    ryy(q_machine=qm,params=weights[8], wires=wires)
    rzz(q_machine=qm,params=weights[9], wires=wires)
    rxx(q_machine=qm,params=weights[10], wires=wires)
    rzx(q_machine=qm,params=weights[11], wires=wires)
    rzx(q_machine=qm,params=weights[12], wires=wires)
    rzz(q_machine=qm,params=weights[13], wires=wires)
    ryy(q_machine=qm,params=weights[14], wires=wires)
    return qm


num_wires = 6

def conv_net(qm, weights, last_layer_weights, features):

    layers = weights.shape[1]
    wires = list(range(num_wires))

    VQC_AmplitudeEmbedding(input_feature = features, q_machine=qm)

    # adds convolutional and pooling layers
    for j in range(layers):
        conv_and_pooling(qm, weights[:, j], wires, skip_first_layer=(not j == 0))
        wires = wires[::2]

    assert last_layer_weights.size == 4 ** (len(wires)) - 1, (
        "The size of the last layer weights vector is incorrect!"
        f" \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
    )
    dense_layer(qm, last_layer_weights, wires)

    return probs(q_state=qm.states, num_wires=qm.num_wires, wires=[0])


def load_digits_data(num_train, num_test, rng):
    """Return training and testing data of digits dataset."""
    digits = datasets.load_digits()
    features, labels = digits.data, digits.target

    # only use first two classes
    features = features[np.where((labels == 0) | (labels == 1))]
    labels = labels[np.where((labels == 0) | (labels == 1))]

    # normalize data
    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))

    # subsample train and test split
    train_indices = rng.choice(len(labels), num_train, replace=False)
    test_indices = rng.choice(
        np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False
    )

    x_train, y_train = features[train_indices], labels[train_indices]
    x_test, y_test = features[test_indices], labels[test_indices]

    return x_train, y_train,x_test, y_test


class Qcnn_ising(Module):

    def __init__(self):
        super(Qcnn_ising, self).__init__()
        self.conv = conv_net
        self.qm = QMachine(num_wires,dtype=kcomplex128)
        self.weights = Parameter((18, 2), dtype=kfloat64)
        self.weights_last = Parameter((4 ** 2 -1,1), dtype=kfloat64)

    def forward(self, input):
        self.qm.reset_states(input.shape[0])
        return self.conv(self.qm, self.weights, self.weights_last, input)


from tqdm import tqdm


def train_qcnn(n_train, n_test, n_epochs):

    # load data
    x_train, y_train, x_test, y_test = load_digits_data(n_train, n_test, rng)

    # init weights and optimizer
    model = Qcnn_ising()

    opti = Adam(model.parameters(), lr=0.01)

    # data containers
    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

    for step in range(n_epochs):
        model.train()
        opti.zero_grad()

        result = model(QTensor(x_train))

        train_cost = 1.0 - tensor.sums(result[tensor.arange(0, len(y_train)), y_train]) / len(y_train)

        train_cost.backward()
        opti.step()

        train_cost_epochs.append(train_cost.to_numpy())
        # compute accuracy on training data

        train_acc = tensor.sums(result[tensor.arange(0, len(y_train)), y_train] > 0.5) / result.shape[0]

        train_acc_epochs.append(train_acc.to_numpy())

        # compute accuracy and cost on testing data
        test_out = model(QTensor(x_test))
        test_acc = tensor.sums(test_out[tensor.arange(0, len(y_test)), y_test] > 0.5) / test_out.shape[0]
        test_acc_epochs.append(test_acc.to_numpy())
        test_cost = 1.0 - tensor.sums(test_out[tensor.arange(0, len(y_test)), y_test]) / len(y_test)
        test_cost_epochs.append(test_cost.to_numpy())



    return dict(
        n_train=[n_train] * n_epochs,
        step=np.arange(1, n_epochs + 1, dtype=int),
        train_cost=train_cost_epochs,
        train_acc=train_acc_epochs,
        test_cost=test_cost_epochs,
        test_acc=test_acc_epochs,
    )



def run_iterations(n_train):
    results_df = pd.DataFrame(
        columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
    )

    for _ in tqdm(range(n_reps)):
        results = train_qcnn(n_train=n_train, n_test=n_test, n_epochs=n_epochs)

        results_df = pd.concat(
            [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
        )

    return results_df

# run training for multiple sizes
train_sizes = [2, 5, 10, 20, 40, 80]
results_df = run_iterations(n_train=2)


for n_train in train_sizes[1:]:
    results_df = pd.concat([results_df, run_iterations(n_train=n_train)])

save = 1 # 保存数据
draw = 1 # 绘图

if save:
    results_df.to_csv('test_qcnn.csv', index=False)
import pickle

if draw:
    # aggregate dataframe
    results_df = pd.read_csv('test_qcnn.csv')
    df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
    df_agg = df_agg.reset_index()

    sns.set_style('whitegrid')
    colors = sns.color_palette()
    fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

    generalization_errors = []

    # plot losses and accuracies
    for i, n_train in enumerate(train_sizes):
        df = df_agg[df_agg.n_train == n_train]

        dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
        lines = ["o-", "x--", "o-", "x--"]
        labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
        axs = [0, 0, 2, 2]

        for k in range(4):
            ax = axes[axs[k]]
            ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

        # plot final loss difference
        dif = (df[df.step == n_epochs].test_cost["mean"] - df[df.step == n_epochs].train_cost["mean"]).item()
        generalization_errors.append(dif)

    # format loss plot
    ax = axes[0]
    ax.set_title('Train and Test Losses', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # format generalization error plot
    ax = axes[1]
    ax.plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
    ax.set_xscale('log')
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_sizes)
    ax.set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
    ax.set_xlabel('Training Set Size')

    # format loss plot
    ax = axes[2]
    ax.set_title('Train and Test Accuracies', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.5, 1.05)

    legend_elements = [
                            mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
                        ] + [
                            mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
                            mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
                        ]

    axes[0].legend(handles=legend_elements, ncol=3)
    axes[2].legend(handles=legend_elements, ncol=3)

    if all(g > 0 for g in generalization_errors):
        axes[1].set_yscale('log', base=2)
    plt.tight_layout()
    fig.savefig('qcnn_results.png', dpi=150, bbox_inches='tight')
    plt.show()