"""
Hybrid Quantum-Classical Model
Strict SINGLE quantum circuit initialization (Option A).
Fully optimized for training performance.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import pennylane as qml
from typing import Tuple, Dict

from src.models.classical.gnn_baseline import MessagePassingLayer
from src.models.quantum.quantum_circuits import VariationalQuantumCircuit


# ================================================================
#  GLOBAL: Single Quantum Circuit Instance
# ================================================================
_GLOBAL_VQC_CACHE = {}


def get_global_qnode(n_qubits, n_layers, feature_map, entanglement):
    """Returns globally cached QNode."""
    key = (n_qubits, n_layers, feature_map, entanglement)

    if key not in _GLOBAL_VQC_CACHE:
        print("\n⚛️ Creating GLOBAL Quantum Circuit (only once)...")

        vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            entanglement=entanglement,
            backend="default.qubit"
        )

        qnode = vqc.create_circuit()
        _GLOBAL_VQC_CACHE[key] = (vqc, qnode)

    return _GLOBAL_VQC_CACHE[key]


# ================================================================
#  QUANTUM LAYER (Flax Module)
# ================================================================
class QuantumLayer(nn.Module):
    n_qubits: int = 4
    n_layers: int = 2
    feature_map: str = "angle"
    entanglement: str = "linear"

    def setup(self):
        self.vqc, self.qnode = get_global_qnode(
            self.n_qubits,
            self.n_layers,
            self.feature_map,
            self.entanglement
        )
        self.n_params = self.vqc.n_params

    @nn.compact
    def __call__(self, features, training=False):
        q_params = self.param(
            "quantum_params",
            nn.initializers.uniform(scale=np.pi),
            (self.n_layers, 2 * self.n_qubits)
        )
        out = self.qnode(features, q_params)
        return jnp.array(out)


# ================================================================
#  HYBRID QUANTUM-CLASSICAL MODEL (FLAX)
# ================================================================
class HybridQuantumClassicalModel(nn.Module):
    gnn_hidden_dim: int = 128
    gnn_layers: int = 2

    n_qubits: int = 4
    quantum_layers: int = 2

    decoder_hidden_dims: Tuple[int, ...] = (64, 32)
    output_dim: int = 1

    pooling: str = "mean"

    @nn.compact
    def __call__(self, node_features, edge_index, edge_features, training=True):

        # 1. GNN encoder
        x = nn.Dense(self.gnn_hidden_dim)(node_features)
        x = nn.gelu(x)

        for i in range(self.gnn_layers):
            x = MessagePassingLayer(
                hidden_dim=self.gnn_hidden_dim,
                name=f"mp_layer_{i}"
            )(x, edge_index, edge_features, training=training)

        # Pooling
        if self.pooling == "mean":
            graph_embedding = jnp.mean(x, axis=0)
        elif self.pooling == "sum":
            graph_embedding = jnp.sum(x, axis=0)
        else:
            graph_embedding = jnp.max(x, axis=0)

        # 2. Compress for quantum input
        compressed = nn.Dense(self.n_qubits)(graph_embedding)
        compressed = nn.tanh(compressed)
        q_input = (compressed + 1.0) * (np.pi / 2)

        # 3. Quantum layer
        q_out = QuantumLayer(
            n_qubits=self.n_qubits,
            n_layers=self.quantum_layers
        )(q_input, training=training)

        # 4. Decoder MLP
        out = jnp.concatenate([graph_embedding, q_out])

        for i, dim in enumerate(self.decoder_hidden_dims):
            out = nn.Dense(dim)(out)
            out = nn.gelu(out)
            out = nn.Dropout(0.10)(out, deterministic=not training)

        out = nn.Dense(self.output_dim)(out)
        return out.squeeze()


# ================================================================
#  PREDICTOR WRAPPER
# ================================================================
class HybridRegressor:
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        n_qubits: int = 4,
        quantum_layers: int = 2,
        decoder_hidden_dims: Tuple[int, ...] = (64, 32),
        output_dim: int = 1,
        seed: int = 42
    ):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.rng = jax.random.PRNGKey(seed)

        self.model = HybridQuantumClassicalModel(
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_layers=gnn_layers,
            n_qubits=n_qubits,
            quantum_layers=quantum_layers,
            decoder_hidden_dims=decoder_hidden_dims,
            output_dim=output_dim
        )

        print("✅ Hybrid Model initialized:")

    def init_params(self, dummy_graph):
        self.rng, init_rng = jax.random.split(self.rng)

        self.params = self.model.init(
            init_rng,
            node_features=dummy_graph["node_features"],
            edge_index=dummy_graph["edge_index"],
            edge_features=dummy_graph["edge_features"],
            training=False
        )

        return self.params

    @staticmethod
    def create_dummy_graph(node_feat_dim, edge_feat_dim):
        return {
            "node_features": jnp.zeros((10, node_feat_dim)),
            "edge_index": jnp.array([[0, 1, 2], [1, 2, 0]]),
            "edge_features": jnp.zeros((3, edge_feat_dim))
        }

    def forward(self, params, graph, training=False):
        # Use a fresh dropout key per call during training
        dropout_rngs = None
        if training:
            self.rng, dk = jax.random.split(self.rng)
            dropout_rngs = {"dropout": dk}

        return self.model.apply(
            params,
            node_features=graph["node_features"],
            edge_index=graph["edge_index"],
            edge_features=graph["edge_features"],
            training=training,
            rngs=dropout_rngs
        )

    def predict(self, params, graph):
        return self.forward(params, graph, training=False)

    def count_parameters(self):
        """
        Count total trainable parameters in the hybrid model.
        Uses current params if available, otherwise initializes a dummy tree.
        """
        params = getattr(self, "params", None)
        if params is None:
            dummy = self.create_dummy_graph(self.node_feat_dim, self.edge_feat_dim)
            _, init_rng = jax.random.split(self.rng)
            params = self.model.init(
                init_rng,
                node_features=dummy["node_features"],
                edge_index=dummy["edge_index"],
                edge_features=dummy["edge_features"],
                training=False
            )
        leaves, _ = jax.tree_util.tree_flatten(params)
        return int(sum(np.prod(p.shape) for p in leaves))


# ================================================================
# TEST
# ================================================================
def test_hybrid_model():
    print("🧪 Testing Hybrid Model...")

    dummy = HybridRegressor.create_dummy_graph(45, 6)
    predictor = HybridRegressor(
        node_feat_dim=45,
        edge_feat_dim=6,
        gnn_hidden_dim=64
    )

    params = predictor.init_params(dummy)
    pred = predictor.predict(params, dummy)
    print("Prediction:", float(pred))


if __name__ == "__main__":
    test_hybrid_model()
