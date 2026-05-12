import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from tqdm import tqdm

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
n_reps = 5
n_episodes_per_epoch = 20  # 每个epoch的任务数
n_test_episodes = 20       # 测试的任务数
n_epochs = 10
n_way = 5                    # 每个任务的类别数（C-way）
n_query = 3                  # 每个类别的查询样本数

# 完全复用你提供的量子层定义
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

# 修改后的量子嵌入网络：输出嵌入向量而非分类概率
def conv_embed(qm, weights, last_layer_weights, features):
    layers = weights.shape[1]
    wires = list(range(num_wires))
    # 振幅编码输入特征，和原QCNN保持一致
    qm.reset_states(features.shape[0])
    VQC_AmplitudeEmbedding(input_feature = features, q_machine=qm)
    # 卷积池化层特征提取
    for j in range(layers):
        conv_and_pooling(qm, weights[:, j], wires, skip_first_layer=(not j == 0))
        wires = wires[::2]
    # 全连接层
    dense_layer(qm, last_layer_weights, wires)
    # 输出所有剩余比特的测量概率作为嵌入向量
    return probs(q_state=qm.states, num_wires=qm.num_wires, wires=wires)

# 加载所有类别的手写数字数据，用于少样本任务采样
def load_digits_data(rng):
    digits = datasets.load_digits()
    features, labels = digits.data, digits.target
    # 和原QCNN一致的归一化，适配振幅编码
    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))
    # 按类别分组，方便后续任务采样
    class_data = {}
    for c in np.unique(labels):
        class_data[c] = features[labels == c]
    return class_data

# 采样单个少样本任务（episode）
def sample_episode(class_data, n_way, n_shot, n_query, rng):
    # 随机选择n_way个类别
    classes = rng.choice(list(class_data.keys()), n_way, replace=False)
    support_x, query_x, query_y = [], [], []
    for idx, c in enumerate(classes):
        # 每个类别采样n_shot + n_query个样本
        samples = rng.choice(class_data[c], n_shot + n_query, replace=False)
        support_x.append(samples[:n_shot])       # 支持集样本
        query_x.append(samples[n_shot:])          # 查询集样本
        query_y.extend([idx] * n_query)           # 查询集标签（当前任务内的类别索引）
    return np.array(support_x), np.array(query_x), np.array(query_y)

# 原型网络模型定义
class QPrototypicalNet(Module):
    def __init__(self):
        super(QPrototypicalNet, self).__init__()
        self.embed_net = conv_embed
        self.qm = QMachine(num_wires,dtype=kcomplex128)
        # 和原QCNN完全一致的可训练参数，复用量子电路结构
        self.weights = Parameter((18, 2), dtype=kfloat64)
        self.weights_last = Parameter((4 ** 2 -1,1), dtype=kfloat64)

    def forward(self, support_x, query_x, query_y):
        # 1. 处理支持集：提取嵌入并计算类别原型
        n_way, n_shot, _ = support_x.shape
        support_flat = support_x.reshape((-1, support_x.shape[-1]))
        support_emb = self.embed_net(self.qm, self.weights, self.weights_last, support_flat)
        support_emb = support_emb.reshape((n_way, n_shot, -1))
        # 计算每个类别的原型：支持集嵌入的均值
        prototypes = tensor.mean(support_emb, axis=1)  # [n_way, embed_dim]

        # 2. 处理查询集：提取嵌入
        n_way_q, n_query_q, _ = query_x.shape
        query_flat = query_x.reshape((-1, query_x.shape[-1]))
        query_emb = self.embed_net(self.qm, self.weights, self.weights_last, query_flat)  # [n_query_total, embed_dim]

        # 3. 计算查询样本到原型的欧氏距离
        query_emb = query_emb.unsqueeze(1)  # [Nq, 1, d]
        prototypes = prototypes.unsqueeze(0) # [1, Nc, d]
        diffs = query_emb - prototypes       # 广播计算差值 [Nq, Nc, d]
        distances = tensor.sums(diffs ** 2, axis=2)  # [Nq, Nc] 欧氏距离平方

        # 4. 计算交叉熵损失
        logits = -distances  # 距离越小，分类得分越高
        # 数值稳定的softmax
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = tensor.exp(logits - max_logits)
        sum_exp = tensor.sums(exp_logits, axis=1, keepdims=True)
        softmax = exp_logits / sum_exp

        # 计算损失
        n_q = len(query_y)
        loss = -tensor.sums(tensor.log(softmax[tensor.arange(0, n_q), QTensor(query_y)])) / n_q
        # 计算准确率
        preds = tensor.argmax(softmax, dim=1)
        acc = tensor.sums(preds == QTensor(query_y)) / n_q

        return loss, acc

# 训练函数
def train_protonet(n_shot, n_epochs):
    # 加载数据
    class_data = load_digits_data(rng)
    # 划分训练/测试类别（6个训练类，4个测试类，模拟 unseen 类别测试）
    all_classes = list(class_data.keys())
    rng.shuffle(all_classes)
    train_classes = {c: class_data[c] for c in all_classes[:5]}
    test_classes = {c: class_data[c] for c in all_classes[5:]}

    # 初始化模型和优化器
    model = QPrototypicalNet()
    opti = Adam(model.parameters(), lr=0.01)

    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

    for step in range(n_epochs):
        model.train()
        # 训练阶段：跑多个训练任务
        train_loss, train_acc = 0.0, 0.0
        for _ in range(n_episodes_per_epoch):
            opti.zero_grad()
            # 采样训练任务
            s_x, q_x, q_y = sample_episode(train_classes, n_way, n_shot, n_query, rng)
            # 前向传播
            loss, acc = model(s_x, q_x, q_y)
            # 反向传播
            loss.backward()
            opti.step()
            train_loss += loss.to_numpy()
            train_acc += acc.to_numpy()
        # 平均训练指标
        train_loss /= n_episodes_per_epoch
        train_acc /= n_episodes_per_epoch
        train_cost_epochs.append(train_loss)
        train_acc_epochs.append(train_acc)

        # 测试阶段：跑多个测试任务
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        for _ in range(n_test_episodes):
            s_x, q_x, q_y = sample_episode(test_classes, n_way, n_shot, n_query, rng)
            loss, acc = model(s_x, q_x, q_y)
            test_loss += loss.to_numpy()
            test_acc += acc.to_numpy()
        test_loss /= n_test_episodes
        test_acc /= n_test_episodes
        test_cost_epochs.append(test_loss)
        test_acc_epochs.append(test_acc)

    return dict(
        n_shot=[n_shot] * n_epochs,
        step=np.arange(1, n_epochs + 1, dtype=int),
        train_cost=train_cost_epochs,
        train_acc=train_acc_epochs,
        test_cost=test_cost_epochs,
        test_acc=test_acc_epochs,
    )

# 多次重复实验，统计结果
def run_iterations(n_shot):
    results_df = pd.DataFrame(
        columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_shot"]
    )
    for _ in tqdm(range(n_reps)):
        results = train_protonet(n_shot=n_shot, n_epochs=n_epochs)
        results_df = pd.concat(
            [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
        )
    return results_df

# 运行不同shot数的实验
shot_sizes = [1, 2, 5, 10, 20, 30]
results_df = run_iterations(n_shot=1)

for n_shot in shot_sizes[1:]:
    results_df = pd.concat([results_df, run_iterations(n_shot=n_shot)])

save = 1 # 保存数据
draw = 1 # 绘图

if save:
    results_df.to_csv('test_qprotonet.csv', index=False)

if draw:
    # 聚合结果
    results_df = pd.read_csv('test_qprotonet.csv')
    df_agg = results_df.groupby(["n_shot", "step"]).agg(["mean", "std"])
    df_agg = df_agg.reset_index()

    sns.set_style('whitegrid')
    colors = sns.color_palette()
    fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

    generalization_errors = []

    # 绘制损失和准确率
    for i, n_shot in enumerate(shot_sizes):
        df = df_agg[df_agg.n_shot == n_shot]
        dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
        lines = ["o-", "x--", "o-", "x--"]
        labels = [fr"$K={n_shot}$", None, fr"$K={n_shot}$", None]
        axs = [0, 0, 2, 2]

        for k in range(4):
            ax = axes[axs[k]]
            ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

        # 计算最终泛化误差
        dif = (df[df.step == n_epochs].test_cost["mean"] - df[df.step == n_epochs].train_cost["mean"]).item()
        generalization_errors.append(dif)

    # 损失图
    ax = axes[0]
    ax.set_title('Train and Test Losses', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # 泛化误差图
    ax = axes[1]
    ax.plot(shot_sizes, generalization_errors, "o-", label=r"$gen(K)$")
    ax.set_xscale('log')
    ax.set_xticks(shot_sizes)
    ax.set_xticklabels(shot_sizes)
    ax.set_title(r'Generalization Error $gen(K) = R(K) - \hat{R}_K(K)$', fontsize=14)
    ax.set_xlabel('Support Set Size (K-shot)')

    # 准确率图
    ax = axes[2]
    ax.set_title('Train and Test Accuracies', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.2, 1.05)

    # 图例
    legend_elements = [
                            mpl.lines.Line2D([0], [0], label=f'K={n}', color=colors[i]) for i, n in enumerate(shot_sizes)
                        ] + [
                            mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
                            mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
                        ]

    axes[0].legend(handles=legend_elements, ncol=3)
    axes[2].legend(handles=legend_elements, ncol=3)

    if all(g > 0 for g in generalization_errors):
        axes[1].set_yscale('log', base=2)
    plt.tight_layout()
    fig.savefig('qprotonet_results.png', dpi=150, bbox_inches='tight')
    plt.show()