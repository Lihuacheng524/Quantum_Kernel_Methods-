# train.py 执行流程详解

## 目录

- [1. 文件依赖关系](#1-文件依赖关系)
- [2. 整体流程概览](#2-整体流程概览)
- [3. 详细执行步骤](#3-详细执行步骤)
- [4. 数据流分析](#4-数据流分析)
- [5. 各模型分支详解](#5-各模型分支详解)
- [6. 输出文件说明](#6-输出文件说明)

---

## 1. 文件依赖关系

```
train.py（主入口）
│
├── dataset.py              # 数据加载（MNIST / Omniglot）
│   └── 依赖: sklearn, torch, torchvision
│
├── kernels/
│   ├── zz_overlap.py       # 量子重叠核（pyvqnet 实现）
│   ├── zz_project.py       # 投影量子核（pyvqnet 实现）
│   ├── angle.py            # 角度编码核（pyqpanda3 实现）
│   ├── amplitude.py        # 振幅编码核（pyqpanda3 实现）
│   └── qcnn.py             # 量子卷积神经网络（pyvqnet + pyqpanda3）
│
└── protonet.py             # 经典原型网络基线（纯 numpy + sklearn）
```

---

## 2. 整体流程概览

```
main()
  │
  ├── 循环: datasets_list (["mnist"])
  │     │
  │     ├── 循环: models (6种方法)
  │     │     │
  │     │     ├── 循环: train_sizes ([2, 5, 10, 20, 40, 80])
  │     │     │     │
  │     │     │     └── run_experiment(n_train, model_type, dataset_name)
  │     │     │           │
  │     │     │           ├── load_dataset()           → 加载完整数据集
  │     │     │           ├── loop: cfg.N_REPS (10次)
  │     │     │           │     ├── train_test_split_for_size()  → 划分训练/测试
  │     │     │           │     ├── PCA降维 + MinMax缩放        → 数据预处理
  │     │     │           │     └── 模型分支（6选1）              → 训练+评估
  │     │     │           └── 返回 results 列表
  │     │     │
  │     │     └── (收集完单个模型在6个size上的全部结果)
  │     │
  │     └── (收集完6个模型在全部size上的结果)
  │
  ├── 保存 CSV: experiment_results.csv
  ├── 绘制对比图: mnist_performance.png
  └── 输出 LaTeX 表格: (终端打印)
```

---

## 3. 详细执行步骤

### 3.1 入口：`main()`（第 146 行）

```python
def main():
    datasets_list = ["mnist"]                     # 当前只跑 MNIST
    train_sizes = [2, 5, 10, 20, 40, 80]          # 6种训练集规模
    models = ["zz_overlap", "zz_projected",        # 6种对比方法
              "amplitude", "angle", "qcnn", "protonet"]
```

### 3.2 实验调度（第 152~158 行）

三层嵌套循环：

```
外层:  数据集循环        → 1×（当前只有 mnist）
中层:  模型循环          → 6×
内层:  训练集大小循环      → 6×
每轮:  run_experiment()  → 10次重复
─────────────────────────────────────
总计: 1 × 6 × 6 × 10 = 360 次独立实验
```

### 3.3 `run_experiment(n_train, model_type, dataset_name)`（第 100 行）

每次调用的流程：

#### 步骤 A — 加载数据

```python
features, labels = load_dataset(dataset_name)
```

| 数据集 | 来源 | 二分类方式 | 原始维度 |
|--------|------|-----------|:--------:|
| `mnist` | `sklearn.datasets.load_digits()` | 只取类别 0 和 1 | 64 维 |
| `omniglot` | `torchvision.datasets.Omniglot()` | 随机选 2 个类别 | 784 维 (28×28) |

预处理：**L2 归一化**（每张图片除以其欧氏范数）

#### 步骤 B — 训练/测试划分（10 次重复）

```python
train_test_split_for_size(n_train, cfg.N_TEST, rng, features, labels)
```

**平衡采样策略**：确保训练集中同时包含两个类别

```
类别 0 取: max(1, n_train // 2) 个样本
类别 1 取: max(1, n_train // 2) 个样本
合并 → shuffle → 取前 n_train 个
```

| n_train | 类别 0 取 | 类别 1 取 | 最终训练集 |
|:-------:|:---------:|:---------:|:---------:|
| 2 | 1 | 1 | 2 |
| 5 | 2 | 2 + 补 1 | 5 |
| 10+ | 各 n//2 | 各 n//2 | n |

测试集始终从剩余样本中取 10 个。

#### 步骤 C — 数据预处理（PCA + MinMax）

```python
n_components = min(cfg.N_QUBITS, x_train.shape[0], x_train.shape[1])
pca = PCA(n_components=n_components).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

samples = np.append(x_train_pca, x_test_pca, axis=0)
minmax = MinMaxScaler((-1, 1)).fit(samples)
x_train_scaled = minmax.transform(x_train_pca)
x_test_scaled = minmax.transform(x_test_pca)
```

- **PCA** 将 64 维原始数据降到 `n_components` 维（≤ 4 qubits）
- **MinMaxScaler** 将所有特征缩放到 [-1, 1]，适配角度编码/振幅编码

#### 步骤 D — 模型分支（6 选 1）

详见第 [5 节](#5-各模型分支详解)。

#### 步骤 E — 结果记录

```python
results.append(dict(
    n_train=n_train,           # 训练集大小
    step=1,                     # 核方法无 epoch 概念
    train_acc=train_acc,        # 训练准确率
    test_acc=test_acc,          # 测试准确率
    model_type=model_type,      # 模型名
    dataset=dataset_name        # 数据集名
))
```

特殊处理：`qcnn` 类型的 `step` 记录为 `cfg.N_EPOCHS`。

### 3.4 结果聚合与可视化（第 161 行起）

```python
df_all = pd.DataFrame(all_results)
df_all.to_csv("experiment_results.csv", index=False)
```

聚合方式：按 `(model_type, n_train)` 分组计算 `test_acc` 的均值和标准差，绘制 errorbar 对比图。

---

## 4. 数据流分析

### 4.1 数据形状变换（以 mnist 为例）

```
原始数据:              (360, 64)    # sklearn digits, 二分类过滤后
       │
       ├── L2归一化:    (360, 64)    # 每行除以其 L2 范数
       │
       ├── 采样:       (n_train, 64) # 随机取 2/5/10/20/40/80 个
       │
       ├── PCA降维:    (n_train, ≤4) # min(4, n_train, 64)
       │
       ├── MinMax:     (n_train, ≤4) # 缩放到 [-1, 1]
       │
       ├── 量子核方法:  核矩阵 K (n_train, n_train) → SVM
       │
       └── QCNN:       振幅编码 (n_train) → 6 量子比特电路 → 分类概率
```

### 4.2 量子方法与经典方法的区别

| 方面 | 量子核方法 (zz_overlap/projected/amplitude/angle) | QCNN | ProtoNet |
|------|:-----------------------------------------------:|:----:|:--------:|
| **训练方式** | 无参数训练（固定核）+ SVM | 参数化电路 + 梯度下降 | 无参数（计算均值原型） |
| **量子电路** | 每对样本调用一次 | 整个 batch 批量编码 | 不使用量子电路 |
| **特征维度** | 4 维（PCA 后） | 64 维（振幅编码直接接收） | 4 维（PCA 后） |

---

## 5. 各模型分支详解

### 5.1 `zz_overlap` — 量子重叠核（第 117 行）

- **实现文件**：[kernels/zz_overlap.py](file:///home/lihuacheng/projects/bishe/kernels/zz_overlap.py)
- **核心原理**：重叠测试电路 `K(x1,x2) = |⟨φ(x1)|φ(x2)⟩|²`
- **编码方式**：H → Rz → CRz 纠缠层（角度编码）
- **输出**：Gram 矩阵（numpy 数组）
- **分类器**：`SVC(kernel="precomputed")`

### 5.2 `zz_projected` — 投影量子核（第 122 行）

- **实现文件**：[kernels/zz_project.py](file:///home/lihuacheng/projects/bishe/kernels/zz_project.py)
- **核心原理**：VQC_ZZFeatureMap 编码 → 测量 Pauli X/Y/Z 期望值 → 高斯核 `exp(-γ·||投影(x1)-投影(x2)||²)`
- **特征维度**：3N（N 个比特的 X/Y/Z 期望值各一）
- **输出**：Gram 矩阵（需 `.to_numpy()` 转换）
- **分类器**：`SVC(kernel="precomputed")`

### 5.3 `amplitude` — 振幅编码核（第 127 行）

- **实现文件**：[kernels/amplitude.py](file:///home/lihuacheng/projects/bishe/kernels/amplitude.py)
- **核心原理**：振幅编码 → 重叠测试 `|⟨x1|x2⟩|²`
- **自动补零**：当特征维度 < 2ⁿ_qubits 时自动 pad（如 4 维 → 16 维）
- **后端**：pyqpanda3（经典模拟器 + 精确概率计算）
- **并行**：`joblib.Parallel` 加速
- **分类器**：`SVC(kernel="precomputed")`

### 5.4 `angle` — 角度编码核（第 132 行）

- **实现文件**：[kernels/angle.py](file:///home/lihuacheng/projects/bishe/kernels/angle.py)
- **核心原理**：角度编码 + 可训练参数优化 → 重叠测试
- **可训练**：通过 KTA（Kernel Target Alignment）损失优化角度参数
- **优化器**：`scipy.optimize.minimize(method='L-BFGS-B', maxiter=20)`
- **后端**：pyqpanda3
- **并行**：`joblib.Parallel` 加速
- **分类器**：`SVC(kernel="precomputed")`

### 5.5 `qcnn` — 量子卷积神经网络（第 137 行）

- **实现文件**：[kernels/qcnn.py](file:///home/lihuacheng/projects/bishe/kernels/qcnn.py)
- **核心结构**：
  - 6 量子比特的量子电路
  - 振幅编码（AmplitudeEmbedding）
  - 2 层卷积层（IsingXX/YY/ZZ + U3 门）
  - 2 层池化层（CNOT + U3）
  - 全连接层（RZZ/RXX/RYY/RZX 门）
- **训练方式**：Adam 优化器，10 epoch
- **损失函数**：交叉熵（`1 - sum(正确类概率)/batch`）
- **参数数量**：18×2 + (4²-1)×1 = 36 + 15 = 51 个参数

### 5.6 `protonet` — 经典原型网络基线（第 140 行）

- **实现文件**：[protonet.py](file:///home/lihuacheng/projects/bishe/protonet.py)
- **核心原理**：用训练集均值计算各类原型，欧氏距离分类
- **训练方式**：无参数（非参数化方法）
- **计算量**：极低（只有 numpy 矩阵运算）
- **分类器**：最近质心分类（等同于 1-NN 在原型空间）

---

## 6. 输出文件说明

### 运行命令

```bash
python train.py
```

### 生成文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `experiment_results.csv` | CSV | 所有实验结果，含 6 列（n_train, step, train_acc, test_acc, model_type, dataset） |
| `mnist_performance.png` | PNG | 6 种模型在 mnist 上的测试准确率对比图（含误差棒） |
| 终端输出 | LaTeX | 可直接复制到论文中的 `\begin{table}` 表格 |

### CSV 文件结构

```csv
n_train,step,train_acc,test_acc,model_type,dataset
2,1,0.75,0.5,zz_overlap,mnist
2,1,1.0,0.6,zz_overlap,mnist
...
80,10,0.95,0.9,qcnn,mnist
```

### 对比图说明

- 横轴：训练集大小（对数坐标）
- 纵轴：测试准确率
- 虚线：随机猜测基线（0.5）
- 误差棒：10 次重复实验的标准差
