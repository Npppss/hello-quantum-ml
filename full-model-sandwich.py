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

# Bagian 2 Dataset Dummy & Persiapan Training
# Membuat dataset dummy (misalnya 100 sampel, 4 fitur)
torch.manual_seed(42)  # Agar hasilnya konsisten saat diulang
X_train = torch.rand(100, 4)

# Membuat label biner (0 atau 1) dengan aturan sederhana:
# Jika total nilai dari 4 fitur > 2, maka labelnya 1. Jika tidak, 0.
y_train = (X_train.sum(dim=1) > 2).float().unsqueeze(1)  # Bentuk (100, 1)

model = HybridModel()

# Menggunakan Binary Cross Entropy (BCELoss) yang ideal untuk klasifikasi biner
criterion = nn.BCELoss()
# Optimizer Adam (algoritma klasik) untuk memperbarui seluruh bobot
optimizer = optim.Adam(model.parameters(), lr=0.1)