import numpy as np
from joblib import Parallel, delayed

PSD_REG = 1e-5

def _safe_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm > 0:
        return (x / norm).tolist()
    return np.zeros_like(x).tolist()


def build_amplitude_kernel(X_1, X_2=None, n_qubits=4):
    if X_2 is None:
        X_2 = X_1
    assert X_1.shape[1] == X_2.shape[1], "Feature dimension mismatch"

    required_dim = 2 ** n_qubits
    if X_1.shape[1] < required_dim:
        X_1 = np.pad(X_1, ((0, 0), (0, required_dim - X_1.shape[1])), mode='constant')
        X_2 = np.pad(X_2, ((0, 0), (0, required_dim - X_2.shape[1])), mode='constant')

    N1, N2 = len(X_1), len(X_2)
    K = np.zeros((N1, N2))

    pairs = [(i,j) for i in range(N1) for j in range(N2)]
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(qpanda_overlap_amplitude_single)(X_1[i], X_2[j], n_qubits)
        for i,j in pairs
    )

    for idx, (i,j) in enumerate(pairs):
        K[i,j] = results[idx]

    if N1 == N2:
        K += PSD_REG * np.eye(N1)
    return K


def qpanda_overlap_amplitude_single(x1, x2, n_qubits):
    import numpy as np
    from pyqpanda3.core import Encode
    x1_norm = _safe_normalize(x1)
    x2_norm = _safe_normalize(x2)
    qubits = list(range(n_qubits))
    enc1 = Encode()
    enc1.amplitude_encode(qubits, x1_norm)
    m1 = np.array(enc1.get_circuit().matrix())
    enc2 = Encode()
    enc2.amplitude_encode(qubits, x2_norm)
    m2 = np.array(enc2.get_circuit().matrix())
    m_total = m2.conj().T @ m1
    return abs(m_total[0, 0]) ** 2
