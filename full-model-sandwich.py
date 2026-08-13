import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# Bagian 1 Model Hibrida
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, n_qubits)
        weight_shapes = {"weights": (3, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc2 = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.qlayer(x)
        x = torch.sigmoid(self.fc2(x))
        return x