import pennylane as qml
import torch
import torch.nn as nn

# inisialisasi perangkat kuantum
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# membuat QNode untuk sirkuit kuantum dengan Pytorch
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Data Encoding: Memasukkan data klasik ke dalam qubit
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Parameterized Quantum Circuit (PQC): Lapisan yang akan dilatih
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    # Pengukuran (Measurement)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Membuat model hybrid kuantum-klasik
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Lapisan klasik 1: Mengompres 4 fitur masukan menjadi 2 fitur (sesuai jumlah qubit)
        self.fcl=nn.Linear(in_features=4, out_features=n_qubits)

        # Mengonfigurasi bentuk bobot (weights) untuk sirkuit kuantum
        # Kita pakai 3 layer entanglement, dengan 2 qubit
        weights_shape={"weights": (3, n_qubits)}

        # Lapisan kuantum: Mengonversi QNode menjadi layer PyTorch
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weights_shape)

        # Lapisan klasik 2: Mengubah 2 nilai ekspektasi kuantum menjadi 1 output prediksi
        self.fcl2 = nn.Linear(in_features=n_qubits, out_features=1)

    def forward(self, x):
        # Aliran Data (Forward Pass)
        x=torch.relu(self.fcl(x)) # Lapisan klasik pertama
        x=self.qlayer(x)   # Lapisan kuantum
        x=torch.sigmoid(self.fcl2(x))            # Lapisan klasik kedua
        return x

# Uji coba model hybrid
model = HybridModel()
print("Arsitektur Model AI Hibrida Kita:\n", model)

# Buat 1 data dummy (1 batch, 4 fitur)
dummy_data = torch.rand(1, 4)
prediksi = model(dummy_data)

print(f"\nHasil Prediksi Awal (Sebelum Training): {prediksi.item():.4f}")