import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

SEED = 0
PSD_REG = 1e-5
ANGLE_N_LAYERS = 2


def build_angle_kernel(X_1, X_2=None, optimal_params=None, X_support=None, y_support=None):
    if X_2 is None:
        X_2 = X_1
    assert X_1.shape[1] == X_2.shape[1], "Feature dimension mismatch"

    if optimal_params is None and X_support is not None and y_support is not None:
        optimal_params = optimize_angle_params(X_support, y_support)

    N1, N2 = len(X_1), len(X_2)
    K = np.zeros((N1, N2))

    pairs = [(i,j) for i in range(N1) for j in range(N2)]
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(qpanda_overlap_angle_single)(X_1[i], X_2[j], optimal_params)
        for i,j in pairs
    )

    for idx, (i,j) in enumerate(pairs):
        K[i,j] = results[idx]

    if N1 == N2:
        K += PSD_REG * np.eye(N1)
    return K


def optimize_angle_params(X_support, y_support, n_layers=ANGLE_N_LAYERS):
    n_qubits = X_support.shape[1]
    np.random.seed(SEED)
    init_params = np.random.uniform(-np.pi/4, np.pi/4, (n_layers, n_qubits))
    y_mat = np.equal.outer(y_support, y_support).astype(float)

    def kta_loss(params_flat):
        params = params_flat.reshape(n_layers, n_qubits)
        N = len(X_support)
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                K[i,j] = K[j,i] = qpanda_overlap_angle_single(X_support[i], X_support[j], params)
        kta = np.sum(K * y_mat) / (np.linalg.norm(K) * np.linalg.norm(y_mat) + 1e-8)
        return -kta

    res = minimize(kta_loss, init_params.flatten(), method='L-BFGS-B', options={'maxiter': 20})
    return res.x.reshape(n_layers, n_qubits)


def angle_feature_map(qubits, x, params):
    import numpy as np
    from pyqpanda3.core import QCircuit, Encode, CNOT, RY
    x_norm = (x / np.linalg.norm(x)).tolist()
    circ = QCircuit()
    enc = Encode()
    enc.angle_encode(qubits, x_norm)
    circ.append(enc.get_circuit())
    for layer in range(params.shape[0]):
        for i in range(len(qubits)-1):
            circ.append(CNOT(qubits[i], qubits[i+1]))
        for i, q in enumerate(qubits):
            circ.append(RY(q, params[layer, i]))
    return circ


def qpanda_overlap_angle_single(x1, x2, params):
    import numpy as np
    n_qubits = len(x1)
    qubits = list(range(n_qubits))
    m1 = np.array(angle_feature_map(qubits, x1, params).matrix())
    m2 = np.array(angle_feature_map(qubits, x2, params).matrix())
    m_total = m2.conj().T @ m1
    return abs(m_total[0, 0]) ** 2
