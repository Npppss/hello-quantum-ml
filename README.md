# Quantum Machine Learning: Variational Quantum Classifier (VQC)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-Machine%20Learning-purple)
![License](https://img.shields.io/badge/License-MIT-green)

A hybrid quantum-classical machine learning implementation using a Variational Quantum Classifier (VQC). This project demonstrates how to build, train, and evaluate a quantum neural network using IBM's Qiskit framework.

## Project Overview

This repository explores the fundamentals of Quantum Machine Learning (QML) by implementing a binary classifier on a synthetic dataset. The model uses a hybrid loop where a quantum processing unit (QPU) handles the feature mapping and parameterized circuits, while a classical CPU optimizes the weights.

### Architecture Highlights:
- **Data Encoding:** `ZZFeatureMap` for mapping classical data into quantum states (angle encoding with entanglement).
- **Parameterized Quantum Circuit (Ansatz):** `RealAmplitudes` using $R_y$ and CNOT gates to act as the trainable weights.
- **Classical Optimizer:** `COBYLA` (Constrained Optimization BY Linear Approximations) for gradient-free optimization of the circuit parameters.

## Installation

To run this project locally, ensure you have Python installed, then install the required dependencies:

```bash
pip install qiskit qiskit-machine-learning scikit-learn matplotlib numpy