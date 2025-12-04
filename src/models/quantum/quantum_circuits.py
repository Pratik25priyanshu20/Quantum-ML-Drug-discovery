#src/models/quantum/quantum_circuits.py
"""
Quantum Circuits for Molecular Property Prediction
Variational Quantum Circuits (VQC) with PennyLane + JAX
"""

import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List


# ================================================================
# 🔹 1. FEATURE MAPS
# ================================================================

class QuantumFeatureMap:

    @staticmethod
    def angle_encoding(features: jnp.ndarray, wires: List[int]):
        for i, w in enumerate(wires):
            qml.RY(features[i], wires=w)

    @staticmethod
    def amplitude_encoding(features: jnp.ndarray, wires: List[int]):
        norm = jnp.linalg.norm(features)
        safe_norm = jnp.where(norm > 1e-8, norm, 1.0)
        qml.QubitStateVector(features / safe_norm, wires=wires)

    @staticmethod
    def iqp_encoding(features: jnp.ndarray, wires: List[int], reps=1):
        for _ in range(reps):
            for w in wires:
                qml.Hadamard(wires=w)
            for i, w in enumerate(wires):
                qml.RZ(features[i], wires=w)
            for i in range(len(wires)-1):
                qml.CNOT(wires=[wires[i], wires[i+1]])


# ================================================================
# 🔹 2. VQC — Variational Quantum Circuit
# ================================================================

class VariationalQuantumCircuit:

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: str = "angle",
        entanglement: str = "linear",
        measurement: str = "z",
        backend: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.measurement = measurement

        # 🟢 Stable device for JAX
        self.dev = qml.device(backend, wires=n_qubits)

        # Total trainable parameters
        self.n_params = n_layers * (2 * n_qubits)

        print(f"✅ Variational Quantum Circuit initialized:")
        print(f"   Qubits: {n_qubits}")
        print(f"   Layers: {n_layers}")
        print(f"   Feature map: {feature_map}")
        print(f"   Entanglement: {entanglement}")
        print(f"   Parameters per layer: {2*n_qubits}")
        print(f"   Total parameters: {self.n_params}")
        print(f"   Backend: {backend}")

    # ----------------------------------------------

    def _encode(self, x):
        wires = list(range(self.n_qubits))

        if self.feature_map == "angle":
            QuantumFeatureMap.angle_encoding(x, wires)

        elif self.feature_map == "amplitude":
            target = 2 ** self.n_qubits
            if len(x) < target:
                x = jnp.pad(x, (0, target - len(x)))
            x = x[:target]
            QuantumFeatureMap.amplitude_encoding(x, wires)

        elif self.feature_map == "iqp":
            QuantumFeatureMap.iqp_encoding(x, wires)

    # ----------------------------------------------

    def _layer(self, params):
        n = self.n_qubits

        # rotations
        for i in range(n):
            qml.RY(params[i], wires=i)
            qml.RZ(params[n + i], wires=i)

        # entanglement
        if self.entanglement == "linear":
            for i in range(n - 1):
                qml.CNOT(wires=[i, i+1])
        elif self.entanglement == "circular":
            for i in range(n - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n-1, 0])
        elif self.entanglement == "full":
            for i in range(n):
                for j in range(i+1, n):
                    qml.CNOT(wires=[i, j])

    # ----------------------------------------------

    def _measure(self):
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    # ----------------------------------------------

    def create_circuit(self):

        @qml.qnode(self.dev, interface="jax")
        def circuit(x, params):
            self._encode(x)
            for l in range(self.n_layers):
                self._layer(params[l])
            return self._measure()

        return circuit

    # ----------------------------------------------

    def init_params(self, seed=42):
        key = jax.random.PRNGKey(seed)
        return jax.random.uniform(
            key,
            shape=(self.n_layers, 2*self.n_qubits),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

    # ----------------------------------------------

    def count_parameters(self):
        return self.n_params


# ================================================================
# 🔹 3. High-level Quantum Neural Network wrapper
# ================================================================

class QuantumNeuralNetwork:
    """
    Simple wrapper that composes a variational circuit with a linear readout.
    Interface is tailored to the training scripts.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        output_dim: int = 1,
        feature_map: str = "angle",
        entanglement: str = "linear",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.output_dim = output_dim

        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            entanglement=entanglement,
        )
        self.circuit = self.vqc.create_circuit()

        self.rng = jax.random.PRNGKey(seed)

    def initialize_parameters(self) -> Dict[str, Any]:
        """
        Initialize trainable parameters for the circuit and a linear readout.
        """
        self.rng, subkey_circ, subkey_readout = jax.random.split(self.rng, 3)
        circuit_params = self.vqc.init_params(seed=int(subkey_circ[0]))

        # Linear readout: maps n_qubits expectation values -> output_dim
        readout_shape = (self.output_dim, self.n_qubits)
        readout = jax.random.normal(subkey_readout, shape=readout_shape) * 0.1
        bias = jnp.zeros((self.output_dim,))

        return {
            "circuit": circuit_params,
            "readout": readout,
            "bias": bias,
        }

    def forward(self, params: Dict[str, Any], features: jnp.ndarray) -> jnp.ndarray:
        """
        Run the quantum circuit and apply linear readout.
        Returns a scalar if output_dim == 1.
        """
        expvals = jnp.asarray(self.circuit(features, params["circuit"]))  # shape (n_qubits,)
        logits = jnp.dot(params["readout"], expvals) + params["bias"]
        return logits.squeeze() if self.output_dim == 1 else logits

    def count_parameters(self) -> int:
        circuit_params = self.vqc.count_parameters()
        readout_params = self.output_dim * (self.n_qubits + 1)  # weights + bias
        return circuit_params + readout_params
