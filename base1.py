import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.linalg import sqrtm, expm
import numpy.linalg as la
from tqdm import tqdm

import sys
sys.path.insert(0, "./")
from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.qnn.vqc.qcircuit import PauliZ, VQC_ZZFeatureMap, PauliX, PauliY, hadamard, crz, rz
from pyvqnet.qnn.vqc import QMachine
from pyvqnet.qnn.vqc.qmeasure import MeasureAll
from pyvqnet import tensor
import functools as ft

# ========================================================
#  实验参数（与 Qcnn.py 保持一致，便于公平对比）
# ========================================================
seed = 0                                   # 随机种子，保证可复现
rng = np.random.default_rng(seed=seed)      # 全局随机数生成器
n_reps = 10                                 # 每个训练集大小的重复实验次数
n_test = 10                                 # 测试集大小（与 Qcnn.py 一致）
n_epochs = 10                               # 用于输出格式对齐，核方法无需 epoch
n_qubits = 4                                # 量子核的比特数（PCA 降维后维度）

# ========================================================
#  量子核函数（基于内积重叠测试电路）
# ========================================================
def custom_data_map_func(x):
    """VQC_ZZFeatureMap 的自定义数据映射函数：
    将输入向量 x 的所有元素连乘，得到一个标量系数。"""
    coeff = x[0] if x.shape[0] == 1 else ft.reduce(lambda m, n: m * n, x)
    return coeff.reshape([1])

def vqnet_quantum_kernel(X_1, X_2=None):
    """量子核方法（重叠测试电路）：
    对每一对样本 (x1, x2) 构建量子电路 U(x1)† U(x2)，
    测量 |0...0⟩ 态上的投影概率作为核值 K(x1, x2)。
    本质：量子态重叠 = |⟨φ(x1)|φ(x2)⟩|²"""
    if X_2 is None:
        X_2 = X_1                            # 当 X_2 未提供时，计算训练核矩阵（Gram 矩阵）
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]                          # 量子比特数 = 特征维度

    # 构造投影算符 |0...0⟩⟨0...0|，用于测量全零态概率
    projector = np.zeros((2**N, 2**N))
    projector[0, 0] = 1
    projector = QTensor(projector, dtype=kcomplex128)

    def kernel(x1, x2):
        """计算单对样本的核值 K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²"""
        qm = QMachine(N, dtype=kcomplex128)
        # 编码 x1：H→Rz→CRz 纠缠层
        for i in range(N):
            hadamard(q_machine=qm, wires=i)
            rz(q_machine=qm, params=QTensor(2 * x1[i], dtype=kfloat64), wires=i)
        for i in range(N):
            for j in range(i + 1, N):
                crz(q_machine=qm, params=QTensor(2 * (np.pi - x1[i]) * (np.pi - x1[j]), dtype=kfloat64), wires=[i, j])
        # 编码 x2 的逆电路（dagger）：CRz† → Rz† → H†
        for i in range(N):
            for j in range(i + 1, N):
                crz(q_machine=qm, params=QTensor(2 * (np.pi - x2[i]) * (np.pi - x2[j]), dtype=kfloat64), wires=[i, j], use_dagger=True)
        for i in range(N):
            rz(q_machine=qm, params=QTensor(2 * x2[i], dtype=kfloat64), wires=i, use_dagger=True)
            hadamard(q_machine=qm, wires=i, use_dagger=True)

        # 计算 ⟨0...0|ψ⟩ ⟨ψ|0...0⟩ = |⟨0...0|U(x1)† U(x2)|0...0⟩|²
        states_1 = qm.states.reshape((1, -1))
        states_1 = tensor.conj(states_1)
        states_2 = qm.states.reshape((-1, 1))
        result = tensor.matmul(tensor.conj(states_1), projector)
        result = tensor.matmul(result, states_2)
        return result.to_numpy()[0][0].real      # 提取实部标量

    # 构建完整核矩阵（双重循环遍历所有样本对）
    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(len(X_1)):
        for j in range(len(X_2)):
            gram[i][j] = kernel(X_1[i], X_2[j])
    return gram


def vqnet_projected_quantum_kernel(X_1, X_2=None, params=QTensor([1.0])):
    """投影量子核方法：
    先将每个样本用量子电路映射为投影特征（Pauli X/Y/Z 期望值），
    再在投影特征空间用高斯核（RBF）计算核值。
    核值 = exp(-γ ||proj(x1) - proj(x2)||²)"""
    if X_2 is None:
        X_2 = X_1
    assert X_1.shape[1] == X_2.shape[1], "The training and testing data must have the same dimensionality"

    def projected_xyz_embedding(X):
        """对每个样本用量子电路提取投影特征：
        返回 [⟨X₀⟩,⟨X₁⟩..., ⟨Y₀⟩,⟨Y₁⟩..., ⟨Z₀⟩,⟨Z₁⟩...] 共 3*N 维特征"""
        N = X.shape[1]

        def proj_feature_map(x):
            """单样本投影：ZZFeatureMap 编码 → 测量所有 X/Y/Z 期望值"""
            qm = QMachine(N, dtype=kcomplex128)
            VQC_ZZFeatureMap(x, qm, data_map_func=custom_data_map_func, entanglement="linear")
            return (
                [MeasureAll(obs={f"X{i}": 1})(qm) for i in range(N)]   # 测量所有比特的 ⟨X⟩
                + [MeasureAll(obs={f"Y{i}": 1})(qm) for i in range(N)] # 测量所有比特的 ⟨Y⟩
                + [MeasureAll(obs={f"Z{i}": 1})(qm) for i in range(N)] # 测量所有比特的 ⟨Z⟩
            )
        return [proj_feature_map(x) for x in X]  # 对批量样本逐行提取投影特征

    # 分别对两个数据集提取投影特征
    X_1_proj = projected_xyz_embedding(QTensor(X_1))
    X_2_proj = projected_xyz_embedding(QTensor(X_2))

    # 在投影特征空间计算高斯核（RBF）
    gamma = params[0]                             # 高斯核带宽参数
    gram = tensor.zeros(shape=[X_1.shape[0], X_2.shape[0]], dtype=kfloat64)
    for i in range(len(X_1_proj)):
        for j in range(len(X_2_proj)):
            result = [a - b for a, b in zip(X_1_proj[i], X_2_proj[j])]  # 特征向量逐元素相减
            result = [a**2 for a in result]                              # 平方差
            value = tensor.exp(-gamma * sum(result).squeeze(0))          # exp(-γ·||diff||²)
            gram[i, j] = value
    return gram


def train_test_kernel_svm(train_gram, train_labels, test_gram, test_labels):
    """使用预计算核矩阵训练 SVM，并返回训练/测试准确率"""
    svm = SVC(kernel="precomputed")                # kernel="precomputed" 表示直接使用核矩阵
    svm.fit(train_gram, train_labels)               # 用训练核矩阵拟合 SVM
    y_predict = svm.predict(test_gram)              # 预测测试集
    train_predict = svm.predict(train_gram)         # 预测训练集
    test_acc = np.sum(test_labels == y_predict) / len(test_labels)       # 测试准确率
    train_acc = np.sum(train_labels == train_predict) / len(train_labels) # 训练准确率
    return train_acc, test_acc


# ========================================================
#  数据加载（与 Qcnn.py 完全一致，保证对比公平性）
# ========================================================
def load_digits_data(num_train, num_test, rng):
    """加载手写数字数据集，仅保留类别 0 和 1（二分类），并进行 L2 归一化"""
    digits = datasets.load_digits()               # 加载 8×8 手写数字图像
    features, labels = digits.data, digits.target
    # 只保留类别 0 和 1 的样本
    features = features[np.where((labels == 0) | (labels == 1))]
    labels = labels[np.where((labels == 0) | (labels == 1))]
    # L2 归一化：每张图片除以其欧氏范数，保证在振幅编码中合法
    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))
    return features, labels


# ========================================================
#  核方法实验运行器
# ========================================================
def run_kernel_experiment(n_train, kernel_type="quantum"):
    """在指定训练集大小下，运行 n_reps 次核方法实验。
    kernel_type: "quantum"（量子重叠核）或 "projected"（投影量子核）"""
    # 仅用于验证数据集中确实有 2 个类别
    _, all_labels = load_digits_data(n_train, n_test, rng)

    results = []
    for _ in range(n_reps):                       # 重复实验以统计均值和标准差
        # 按指定大小随机划分训练/测试集
        x_train, y_train, x_test, y_test = train_test_split_for_size(
            n_train, n_test, rng
        )

        # PCA 降维：64 维原始特征 → n_components 维（≤ n_qubits）
        n_components = min(n_qubits, x_train.shape[0], x_train.shape[1])
        pca = PCA(n_components=n_components).fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        # MinMax 缩放：将所有特征缩放到 [-1, 1] 以适应角度编码
        samples = np.append(x_train_pca, x_test_pca, axis=0)
        minmax = MinMaxScaler((-1, 1)).fit(samples)
        x_train_scaled = minmax.transform(x_train_pca)
        x_test_scaled = minmax.transform(x_test_pca)

        # 根据核类型计算训练核矩阵和测试核矩阵
        if kernel_type == "quantum":
            train_gram = vqnet_quantum_kernel(X_1=x_train_scaled)
            test_gram = vqnet_quantum_kernel(X_1=x_test_scaled, X_2=x_train_scaled)
        elif kernel_type == "projected":
            train_gram = vqnet_projected_quantum_kernel(X_1=x_train_scaled)
            test_gram = vqnet_projected_quantum_kernel(X_1=x_test_scaled, X_2=x_train_scaled)
            train_gram = train_gram.to_numpy()    # QTensor → NumPy
            test_gram = test_gram.to_numpy()

        # 用核矩阵训练 SVM 并评估
        train_acc, test_acc = train_test_kernel_svm(train_gram, y_train, test_gram, y_test)

        results.append(dict(
            n_train=n_train,
            step=1,                              # 核方法无 epoch 概念，统一用 step=1
            train_acc=train_acc,
            test_acc=test_acc,
        ))

    return results


def train_test_split_for_size(n_train, n_test, rng):
    """按指定大小划分训练/测试集，保证两个类别都有样本（平衡采样）"""
    features, labels = load_digits_data(n_train, n_test, rng)
    # 分别获取类别 0 和类别 1 的所有样本索引
    class0_indices = np.where(labels == 0)[0]
    class1_indices = np.where(labels == 1)[0]
    # 从两个类别各取一半样本，保证 SVM 至少有 2 类可训练
    idx0 = rng.choice(class0_indices, max(1, n_train // 2), replace=False)
    idx1 = rng.choice(class1_indices, max(1, n_train // 2), replace=False)
    train_indices = np.concatenate([idx0, idx1])[:n_train]  # 合并后截取所需数量
    rng.shuffle(train_indices)                  # 打乱顺序，避免类别偏向影响
    # 从剩余样本中抽取测试集
    test_indices = rng.choice(
        np.setdiff1d(range(len(labels)), train_indices), n_test, replace=False
    )
    return features[train_indices], labels[train_indices], features[test_indices], labels[test_indices]


# ========================================================
#  主实验流程
# ========================================================
train_sizes = [2, 5, 10, 20, 40, 80]            # 训练集大小序列（与 Qcnn.py 完全一致）

# ---- 实验 1：量子重叠核 SVM ----
print("=== Running Quantum Kernel SVM ===")
all_results = []
for n in tqdm(train_sizes):
    all_results.extend(run_kernel_experiment(n, kernel_type="quantum"))

# ---- 实验 2：投影量子核 SVM ----
print("=== Running Projected Quantum Kernel SVM ===")
all_results_proj = []
for n in tqdm(train_sizes):
    all_results_proj.extend(run_kernel_experiment(n, kernel_type="projected"))

# ========================================================
#  保存实验结果到 CSV
# ========================================================
df_quantum = pd.DataFrame(all_results)             # 量子重叠核结果
df_projected = pd.DataFrame(all_results_proj)       # 投影量子核结果

df_quantum.to_csv("kernel_quantum_results.csv", index=False)
df_projected.to_csv("kernel_projected_results.csv", index=False)

print("Quantum kernel results saved to kernel_quantum_results.csv")
print("Projected kernel results saved to kernel_projected_results.csv")

# ========================================================
#  加载 QCNN 结果并与两种量子核方法对比
# ========================================================
draw = 1
if draw:
    # 尝试加载 QCNN 实验结果（需先运行 Qcnn.py）
    try:
        qcnn_results = pd.read_csv("test_qcnn.csv")
        # 按训练集大小聚合 QCNN 各 epoch 的最终测试准确率
        qcnn_agg = qcnn_results.groupby(["n_train"])["test_acc"].agg(["mean", "std"]).reset_index()
        qcnn_agg.columns = ["n_train", "qcnn_mean", "qcnn_std"]
        has_qcnn = True
    except FileNotFoundError:
        print("Warning: test_qcnn.csv not found. Skipping QCNN comparison.")
        has_qcnn = False

    # 聚合量子重叠核结果
    kernel_quantum_agg = df_quantum.groupby(["n_train"])["test_acc"].agg(["mean", "std"]).reset_index()
    kernel_quantum_agg.columns = ["n_train", "quantum_mean", "quantum_std"]

    # 聚合投影量子核结果
    kernel_proj_agg = df_projected.groupby(["n_train"])["test_acc"].agg(["mean", "std"]).reset_index()
    kernel_proj_agg.columns = ["n_train", "projected_mean", "projected_std"]

    # ===== 绘制 QCNN vs 量子核方法的对比图 =====
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 QCNN 结果曲线（带误差棒）
    if has_qcnn:
        ax.errorbar(
            qcnn_agg.n_train, qcnn_agg.qcnn_mean, yerr=qcnn_agg.qcnn_std,
            fmt="o-", label="QCNN (6 qubits, trainable)", capsize=5, capthick=1.5, lw=2.5
        )

    # 绘制量子重叠核结果曲线（带误差棒）
    ax.errorbar(
        kernel_quantum_agg.n_train, kernel_quantum_agg.quantum_mean,
        yerr=kernel_quantum_agg.quantum_std,
        fmt="s--", label=f"Quantum Kernel SVM ({n_qubits} qubits)", capsize=5, capthick=1.5, lw=2
    )

    # 绘制投影量子核结果曲线（带误差棒）
    ax.errorbar(
        kernel_proj_agg.n_train, kernel_proj_agg.projected_mean,
        yerr=kernel_proj_agg.projected_std,
        fmt="^–.", label=f"Projected Quantum Kernel SVM ({n_qubits} qubits)", capsize=5, capthick=1.5, lw=2
    )

    # 设置图表属性
    ax.set_xlabel("Training Set Size", fontsize=14)
    ax.set_ylabel("Test Accuracy", fontsize=14)
    ax.set_title("QCNN vs Quantum Kernel SVM — Test Accuracy Comparison", fontsize=15)
    ax.set_xscale("log")                           # 训练集大小跨度大，用对数坐标更清晰
    ax.set_xticks(train_sizes)
    ax.set_xticklabels(train_sizes)
    ax.set_ylim(0.3, 1.05)                         # 固定 y 轴范围，便于比较
    ax.legend(fontsize=12, loc="lower right")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random guess")  # 随机猜测基线

    plt.tight_layout()                               # 自动调整子图边距
    fig.savefig("qcnn_vs_kernel_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nComparison plot saved to qcnn_vs_kernel_comparison.png")

    # ===== 打印三种方法的数值对比表格 =====
    print("\n===== Numerical Comparison =====")
    print(f"{'n_train':>8} | {'QCNN (mean±std)':>20} | {'Quantum Kernel (mean±std)':>26} | {'Projected Kernel (mean±std)':>28}")
    print("-" * 90)
    for n in train_sizes:
        # 提取 QCNN 结果（如果存在）
        qcnn_str = f"{qcnn_agg[qcnn_agg.n_train==n].qcnn_mean.values[0]:.3f}±{qcnn_agg[qcnn_agg.n_train==n].qcnn_std.values[0]:.3f}" if has_qcnn else "N/A"
        qk = kernel_quantum_agg[kernel_quantum_agg.n_train == n]   # 量子重叠核
        pk = kernel_proj_agg[kernel_proj_agg.n_train == n]         # 投影量子核
        qk_str = f"{qk.quantum_mean.values[0]:.3f}±{qk.quantum_std.values[0]:.3f}"
        pk_str = f"{pk.projected_mean.values[0]:.3f}±{pk.projected_std.values[0]:.3f}"
        print(f"{n:>8} | {qcnn_str:>20} | {qk_str:>26} | {pk_str:>28}")
