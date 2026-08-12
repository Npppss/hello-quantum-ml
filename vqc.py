import numpy as np
import matplotlib.pyplot as plt

# Komponen Qiskit Machine Learning
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

##Scikit-learn untuk Dataset dan Data Splitting
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

# Persiapkan Dataset
print("Persiapan dataset klasik...")
# Membuat dataset dummy: 100 titik data, 2 fitur (X dan Y), 2 kelas (0 dan 1)