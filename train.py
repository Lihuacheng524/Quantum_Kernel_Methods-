import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.linalg import sqrtm, expm
import numpy.linalg as la
from tqdm import tqdm
from joblib import Parallel, delayed

import sys
sys.path.insert(0, "./")
from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.qnn.vqc.qcircuit import PauliZ, VQC_ZZFeatureMap, PauliX, PauliY, hadamard, crz, rz
from pyvqnet.qnn.vqc import QMachine
from pyvqnet.qnn.vqc.qmeasure import MeasureAll
from pyvqnet import tensor
import functools as ft

from dataset import load_dataset
from kernels.zz_project import vqnet_projected_quantum_kernel
from kernels.zz_overlap import vqnet_quantum_kernel
from kernels.angle import build_angle_kernel, optimize_angle_params
from kernels.amplitude import build_amplitude_kernel
from kernels.qcnn import train_test_qcnn
from protonet import train_test_protonet


class Config:
    SEED = 0
    N_EPOCHS = 10
    N_REPS = 10
    N_TEST = 10
    N_QUBITS = 4
    PSD_REG = 1e-5
    ANGLE_N_LAYERS = 2
    N_WAY = 5
    N_QUERY = 15
    N_EPISODES = 100

cfg = Config()
rng = np.random.default_rng(seed=cfg.SEED)
# -------- 核方法 SVM 工具函数 --------
def train_test_kernel_svm(train_gram, train_labels, test_gram, test_labels):
    svm = SVC(kernel="precomputed")
    svm.fit(train_gram, train_labels)
    y_predict = svm.predict(test_gram)
    train_predict = svm.predict(train_gram)
    test_acc = np.sum(test_labels == y_predict) / len(test_labels)
    train_acc = np.sum(train_labels == train_predict) / len(train_labels)
    return train_acc, test_acc


def train_test_split_for_size(n_train, n_test, rng, features, labels):
    class0_indices = np.where(labels == 0)[0]
    class1_indices = np.where(labels == 1)[0]
    n_total = len(class0_indices) + len(class1_indices)
    n_train = min(n_train, n_total - n_test)
    n_train = max(n_train, 2)
    idx0 = rng.choice(class0_indices, min(max(1, n_train // 2), len(class0_indices)), replace=False)
    idx1 = rng.choice(class1_indices, min(max(1, n_train // 2), len(class1_indices)), replace=False)
    train_indices = np.concatenate([idx0, idx1])[:n_train]
    rng.shuffle(train_indices)
    leftover = np.setdiff1d(range(len(labels)), train_indices)
    actual_n_test = min(n_test, len(leftover))
    if actual_n_test == 0:
        train_indices = rng.choice(range(len(labels)), max(2, n_total - n_test), replace=False)
        rng.shuffle(train_indices)
        leftover = np.setdiff1d(range(len(labels)), train_indices)
        actual_n_test = min(n_test, len(leftover))
    test_indices = rng.choice(leftover, actual_n_test, replace=False)
    return features[train_indices], labels[train_indices], features[test_indices], labels[test_indices]


def run_experiment(n_train, model_type, dataset_name):
    features, labels = load_dataset(dataset_name)
    results = []
    for _ in range(cfg.N_REPS):
        x_train, y_train, x_test, y_test = train_test_split_for_size(
            n_train, cfg.N_TEST, rng, features, labels
        )

        if model_type == "qcnn":
            x_train_model = x_train
            x_test_model = x_test
        else:
            n_components = min(cfg.N_QUBITS, x_train.shape[0], x_train.shape[1])
            pca = PCA(n_components=n_components).fit(x_train)
            x_train_pca = pca.transform(x_train)
            x_test_pca = pca.transform(x_test)
            samples = np.append(x_train_pca, x_test_pca, axis=0)
            minmax = MinMaxScaler((-1, 1)).fit(samples)
            x_train_model = minmax.transform(x_train_pca)
            x_test_model = minmax.transform(x_test_pca)

        if model_type == "zz_overlap":
            train_gram = vqnet_quantum_kernel(X_1=x_train_model)
            test_gram = vqnet_quantum_kernel(X_1=x_test_model, X_2=x_train_model)
            train_acc, test_acc = train_test_kernel_svm(train_gram, y_train, test_gram, y_test)

        elif model_type == "zz_projected":
            train_gram = vqnet_projected_quantum_kernel(X_1=x_train_model).to_numpy()
            test_gram = vqnet_projected_quantum_kernel(X_1=x_test_model, X_2=x_train_model).to_numpy()
            train_acc, test_acc = train_test_kernel_svm(train_gram, y_train, test_gram, y_test)

        elif model_type == "amplitude":
            train_gram = build_amplitude_kernel(x_train_model, n_qubits=cfg.N_QUBITS)
            test_gram = build_amplitude_kernel(x_test_model, x_train_model, n_qubits=cfg.N_QUBITS)
            train_acc, test_acc = train_test_kernel_svm(train_gram, y_train, test_gram, y_test)

        elif model_type == "angle":
            optimal_params = optimize_angle_params(x_train_model, y_train)
            train_gram = build_angle_kernel(x_train_model, optimal_params=optimal_params)
            test_gram = build_angle_kernel(x_test_model, x_train_model, optimal_params=optimal_params)
            train_acc, test_acc = train_test_kernel_svm(train_gram, y_train, test_gram, y_test)

        elif model_type == "qcnn":
            train_acc, test_acc = train_test_qcnn(x_train_model, y_train, x_test_model, y_test, n_epochs=cfg.N_EPOCHS)

        elif model_type == "protonet":
            train_acc, test_acc = train_test_protonet(x_train_model, y_train, x_test_model, y_test)

        else:
            raise ValueError(f"Unsupported model: {model_type}")

        results.append(dict(
            n_train=n_train,
            step=1 if model_type != "qcnn" else cfg.N_EPOCHS,
            train_acc=train_acc,
            test_acc=test_acc,
            model_type=model_type,
            dataset=dataset_name
        ))

    return results


def main():
    # datasets_list = ["mnist", "omniglot"]
    datasets_list = ["omniglot", "mnist"]
    
    train_sizes = [2, 5, 10, 20, 40]
    dataset_models = {
        "mnist": ["amplitude", "qcnn", "zz_overlap", "zz_projected", "angle", "protonet"],
        "omniglot": ["angle","amplitude", "zz_overlap", "zz_projected", "protonet"],
    }
    all_results = []
    for dataset in datasets_list:
        models = dataset_models[dataset]
        print(f"\n=== Running experiments on {dataset} ===")
        for model in models:
            print(f"  Running {model} on {dataset}...")
            for n in tqdm(train_sizes):
                all_results.extend(run_experiment(n, model, dataset))

    df_all = pd.DataFrame(all_results)
    df_all.to_csv("experiment_results.csv", index=False)
    print("\nAll results saved to experiment_results.csv")

    model_styles = {
        "amplitude": {"fmt": "D--", "label": "Amplitude Encoding Kernel (4 qubits)", "color": "#ff7f0e"},
        "qcnn": {"fmt": "o-", "label": "QCNN (6 qubits, trainable)", "color": "#d62728"},
        "zz_overlap": {"fmt": "s--", "label": "ZZ Overlap Kernel (4 qubits)", "color": "#1f77b4"},
        "zz_projected": {"fmt": "^-.", "label": "ZZ Projected Kernel (4 qubits)", "color": "#2ca02c"},
        "angle": {"fmt": "v:", "label": "Trainable Angle Kernel (4 qubits)", "color": "#9467bd"},
        "protonet": {"fmt": "x-", "label": "Prototypical Network (Baseline)", "color": "#8c564b"}
    }

    for dataset in datasets_list:
        df_dataset = df_all[df_all["dataset"] == dataset]
        models = dataset_models[dataset]
        agg = df_dataset.groupby(["model_type", "n_train"])["test_acc"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(12, 7))
        for model in models:
            data = agg[agg["model_type"] == model]
            ax.errorbar(
                data.n_train, data["mean"], yerr=data["std"],
                fmt=model_styles[model]["fmt"],
                label=model_styles[model]["label"],
                color=model_styles[model]["color"],
                capsize=5, capthick=1.5, lw=2
            )

        ax.set_xlabel("Training Set Size", fontsize=14)
        ax.set_ylabel("Test Accuracy", fontsize=14)
        ax.set_title(f"Model Performance Comparison — {dataset.upper()} Dataset", fontsize=16)
        ax.set_xscale("log")
        ax.set_xticks(train_sizes)
        ax.set_xticklabels(train_sizes)
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=12, loc="lower right")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random Guess")

        plt.tight_layout()
        fig.savefig(f"{dataset}_performance.png", dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Plot saved to {dataset}_performance.png")

        print(f"\n===== {dataset.upper()} Dataset Results (mean±std) =====")
        header = "Training Size & " + " & ".join([model_styles[m]["label"] for m in models]) + " \\\\"
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\begin{tabular}{c|" + "c" * len(models) + "}")
        print("\\hline")
        print(header)
        print("\\hline")
        for n in train_sizes:
            row = [f"{n}"]
            for model in models:
                data = agg[(agg["model_type"] == model) & (agg["n_train"] == n)]
                if len(data) > 0:
                    row.append(f"{data['mean'].values[0]:.3f}$\\pm${data['std'].values[0]:.3f}")
                else:
                    row.append("N/A")
            print(" & ".join(row) + " \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print(f"\\caption{{Test accuracy comparison across models on {dataset} dataset}}")
        print(f"\\label{{tab:{dataset}_results}}")
        print("\\end{table}")


if __name__ == "__main__":
    main()
