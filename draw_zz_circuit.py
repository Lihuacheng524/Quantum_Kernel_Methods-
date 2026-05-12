import numpy as np

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def build_zz_overlap_circuit_text(N, x1, x2):
    """Print the ZZ overlap kernel interference circuit as text"""
    print("=" * 80)
    print("ZZ Overlap Kernel Interference Circuit (N={} qubits)".format(N))
    print("=" * 80)
    print("\nSample x1 = {}".format(x1))
    print("Sample x2 = {}".format(x2))
    print("\n" + "=" * 80)
    print("Gate Sequence:")
    print("=" * 80)

    gate_count = 0

    print("\n[Stage 1] Initialize superposition + encode x1 (RZ rotations)")
    print("-" * 70)
    for i in range(N):
        gate_count += 1
        print("  Gate {:3d}: H on q{}".format(gate_count, i))
        gate_count += 1
        print("  Gate {:3d}: RZ(2*x1[{}]) = RZ({:.4f}) on q{}".format(
            gate_count, i, 2 * x1[i], i))

    print("\n[Stage 2] Encode x1 entanglement (CRZ gates)")
    print("-" * 70)
    for i in range(N):
        for j in range(i + 1, N):
            param = 2 * (np.pi - x1[i]) * (np.pi - x1[j])
            gate_count += 1
            print("  Gate {:3d}: CRZ({:.4f}) on control=q{}, target=q{}".format(
                gate_count, param, i, j))

    print("\n[Stage 3] Inverse encode x2 entanglement (CRZ dagger)")
    print("-" * 70)
    for i in range(N):
        for j in range(i + 1, N):
            param = 2 * (np.pi - x2[i]) * (np.pi - x2[j])
            gate_count += 1
            print("  Gate {:3d}: CRZ^dag({:.4f}) on control=q{}, target=q{}".format(
                gate_count, param, i, j))

    print("\n[Stage 4] Inverse encode x2 (RZ dagger + Hadamard)")
    print("-" * 70)
    for i in range(N):
        gate_count += 1
        print("  Gate {:3d}: RZ^dag(2*x2[{}]) = RZ^dag({:.4f}) on q{}".format(
            gate_count, i, 2 * x2[i], i))
        gate_count += 1
        print("  Gate {:3d}: H^dag(q{}) = H(q{})".format(gate_count, i, i))

    print("\n" + "=" * 80)
    print("Total gates: {}".format(gate_count))
    print("  - Hadamard gates: {}".format(2 * N))
    print("  - RZ gates: {}".format(N))
    print("  - CRZ gates: {}".format(N * (N - 1) // 2))
    print("  - RZ^dag gates: {}".format(N))
    print("  - CRZ^dag gates: {}".format(N * (N - 1) // 2))
    print("=" * 80)


def draw_circuit_qiskit(N, x1, x2):
    """Draw the circuit using Qiskit"""
    if not HAS_QISKIT:
        print("\nQiskit not installed. Install with: pip install qiskit")
        return

    qc = QuantumCircuit(N)

    for i in range(N):
        qc.h(i)
        qc.rz(2 * x1[i], i)

    for i in range(N):
        for j in range(i + 1, N):
            param = 2 * (np.pi - x1[i]) * (np.pi - x1[j])
            qc.crz(param, i, j)

    for i in range(N):
        for j in range(i + 1, N):
            param = 2 * (np.pi - x2[i]) * (np.pi - x2[j])
            qc.crz(-param, i, j)

    for i in range(N):
        qc.rz(-2 * x2[i], i)
        qc.h(i)

    print("\n\n")
    print("=" * 80)
    print("Circuit Diagram (Qiskit text output)")
    print("=" * 80)
    print(qc.draw(output='text', fold=100))

    try:
        qc.draw(output='mpl', filename='zz_overlap_circuit.png')
        print("\nCircuit diagram image saved to zz_overlap_circuit.png")
    except Exception as e:
        print("\nCould not save image (may need matplotlib): {}".format(e))


def draw_circuit_latex_tikz(N, x1, x2):
    """Generate LaTeX TikZ code for the circuit using qcircuit package"""
    print("\n\n")
    print("=" * 80)
    print("LaTeX Qcircuit Code")
    print("=" * 80)
    print("\nAdd to your LaTeX preamble: \\usepackage{qcircuit}")
    print("\n```latex")
    print("\\begin{equation*}")
    print("\\Qcircuit @C=1.0em @R=1.2em {")

    for i in range(N):
        row = "  \\lstick{\\ket{0}_{" + str(i) + "}} & \\gate{H} & \\gate{RZ(2x_1^{" + str(i) + "})}"
        for j in range(i + 1, N):
            if i == 0:
                row += " & \\ctrl{" + str(j - i) + "}"
            else:
                row += " & \\targ"
        row += " & \\qw"
        for j in range(i + 1, N):
            if i == 0:
                row += " & \\ctrl{" + str(j - i) + "}"
            else:
                row += " & \\targ"
        row += " & \\gate{RZ^\\dag(2x_2^{" + str(i) + "})} & \\gate{H} & \\qw \\\\"
        print(row)

    print("}")
    print("\\end{equation*}")
    print("```\n")


if __name__ == "__main__":
    N = 4
    x1 = np.array([0.1, 0.2, 0.3, 0.4])
    x2 = np.array([0.15, 0.25, 0.35, 0.45])

    build_zz_overlap_circuit_text(N, x1, x2)

    draw_circuit_qiskit(N, x1, x2)

    draw_circuit_latex_tikz(N, x1, x2)
