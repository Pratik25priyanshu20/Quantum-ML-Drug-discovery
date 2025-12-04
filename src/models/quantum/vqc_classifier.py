#src/models/quantum/vqc_regressor.py

"""
VQCRegressor
A wrapper around QuantumNeuralNetwork for molecular property prediction.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

from src.models.quantum.quantum_circuits import QuantumNeuralNetwork


class VQCRegressor:
    """
    A simple regressor wrapper around the Quantum Neural Network.
    This is what the API expects when importing VQCRegressor.
    """

    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        output_dim=1,
        feature_map="angle",
        entanglement="linear",
        seed=42,
    ):
        self.qnn = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=n_layers,
            output_dim=output_dim,
            feature_map=feature_map,
            entanglement=entanglement,
            seed=seed,
        )

        self.params = self.qnn.initialize_parameters()

        print("✅ VQCRegressor initialized")
        print(f"   Total trainable parameters: {self.qnn.count_parameters()}")

    # --------------------------------------------------------------

    def predict(self, X: Any):
        """
        X must be a 1D vector (single sample) or
        2D array (batch of samples).
        """

        X = np.array(X)

        # Single sample → reshape to batch size 1
        is_single = False
        if X.ndim == 1:
            is_single = True
            X = X.reshape(1, -1)

        preds = []
        for x in X:
            x_jax = jnp.array(x)
            y = self.qnn.forward(self.params, x_jax)
            preds.append(np.array(y))

        preds = np.array(preds)

        return preds[0] if is_single else preds

    # --------------------------------------------------------------

    def set_parameters(self, params_dict):
        """Allow loading parameters from saved model."""
        self.params = params_dict

    def get_parameters(self):
        """Return parameters for saving."""
        return self.params