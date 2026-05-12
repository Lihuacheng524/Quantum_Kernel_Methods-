from pyqpanda3.profiling import draw_circuit_features
from pyqpanda3 import core

circuit = core.QCircuit(2)
circuit << core.H(0) << core.CNOT(0, 1)

# 可视化线路特征（连通性、活性、并行性、纠缠、深度）
draw_circuit_features(circuit, save_fn="circuit_features.png")