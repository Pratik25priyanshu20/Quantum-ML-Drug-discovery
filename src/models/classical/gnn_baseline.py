#src/models/classical/gnn_baseline.py

"""
Graph Neural Network Baseline (JAX/Flax)
FINAL — CHECKPOINT-COMPATIBLE VERSION
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Callable


# ======================================================================
# MESSAGE PASSING LAYER
# ======================================================================

class MessagePassingLayer(nn.Module):
    hidden_dim: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, node_features, edge_index, edge_features, training=True):
        num_nodes = node_features.shape[0]
        src, dst = edge_index  # Edge index: [2, num_edges]

        # Source node feature lookup
        src_feat = node_features[src]

        # Message MLP input = concat(node_src, edge_features)
        msg_input = jnp.concatenate([src_feat, edge_features], axis=-1)

        # Message MLP
        msg = nn.Dense(self.hidden_dim)(msg_input)
        msg = self.activation(msg)
        msg = nn.Dense(self.hidden_dim)(msg)

        # Aggregate messages
        agg = jnp.zeros((num_nodes, self.hidden_dim))
        agg = agg.at[dst].add(msg)

        # Update input = concat(node_features, aggregated)
        upd_input = jnp.concatenate([node_features, agg], axis=-1)

        h = nn.Dense(self.hidden_dim)(upd_input)
        h = nn.LayerNorm()(h)
        h = self.activation(h)
        h = nn.Dropout(0.1)(h, deterministic=not training)

        # Residual if same dim
        if node_features.shape[-1] == self.hidden_dim:
            h = h + node_features

        return h


# ======================================================================
# FULL GNN
# ======================================================================

class GraphNeuralNetwork(nn.Module):
    hidden_dim: int = 128
    num_layers: int = 3
    output_dim: int = 1
    pooling: str = "sum"

    @nn.compact
    def __call__(self, node_features, edge_index, edge_features, training=True):

        # Input projection
        x = nn.Dense(self.hidden_dim)(node_features)
        x = nn.gelu(x)

        # Message passing stack
        for i in range(self.num_layers):
            x = MessagePassingLayer(self.hidden_dim, name=f"mp_{i}")(
                x, edge_index, edge_features, training=training
            )

        # Graph pooling
        if self.pooling == "sum":
            g = jnp.sum(x, axis=0)
        elif self.pooling == "mean":
            g = jnp.mean(x, axis=0)
        else:
            g = jnp.max(x, axis=0)

        # Prediction MLP
        g = nn.Dense(256)(g)
        g = nn.gelu(g)
        g = nn.Dense(128)(g)
        g = nn.gelu(g)

        out = nn.Dense(self.output_dim)(g)
        return out.squeeze()


# ======================================================================
# PREDICTOR WRAPPER
# ======================================================================

class GNNPredictor:
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidden_dim=128,
        num_layers=3,
        output_dim=1,
        seed=42
    ):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.model = GraphNeuralNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim
        )

        self.rng = jax.random.PRNGKey(seed)
        self.params = None

        print("✅ GNNPredictor initialized")

    @staticmethod
    def create_dummy_graph(node_dim, edge_dim):
        return {
            "node_features": jnp.zeros((10, node_dim)),
            "edge_index": jnp.array([[0, 1, 2], [1, 2, 0]]),
            "edge_features": jnp.zeros((3, edge_dim)),
        }

    def init_params(self, dummy_graph):
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.model.init(
            init_rng,
            node_features=dummy_graph["node_features"],
            edge_index=dummy_graph["edge_index"],
            edge_features=dummy_graph["edge_features"],
            training=False
        )
        print(f"   Total parameters: {self.count_parameters():,}")
        return self.params

    def forward(self, params, graph, training=False, rngs=None):
        return self.model.apply(
            params,
            node_features=graph["node_features"],
            edge_index=graph["edge_index"],
            edge_features=graph["edge_features"],
            training=training,
            rngs=rngs or {}
        )

    def predict(self, params, graph):
        return self.forward(params, graph, training=False, rngs={})

    def count_parameters(self):
        return sum(x.size for x in jax.tree_util.tree_leaves(self.params))


__all__ = [
    "GNNPredictor",
    "GraphNeuralNetwork",
    "MessagePassingLayer"
]