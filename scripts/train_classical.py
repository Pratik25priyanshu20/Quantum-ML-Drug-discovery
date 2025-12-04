# train_classical.py


import sys
sys.path.append(".")

import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import argparse
import numpy as np

from src.data.loaders import QM9Loader, DataSplitter
from src.data.molecular_features import MolecularFeaturizer
from src.models.classical.gnn_baseline import GNNPredictor
from src.training.trainer import GNNTrainer


# ============================================================
# 📦 STEP 1 — DATA PREPARATION
# ============================================================
def prepare_data(target_property="mu", max_samples=None,
                val_ratio=0.1, test_ratio=0.1):

    print("=" * 70)
    print("📦 STEP 1: DATA PREPARATION")
    print("=" * 70)

    loader = QM9Loader()
    dataset = loader.load(
        target_property=target_property,
        max_samples=max_samples,
    )

    # Split dataset
    train_ds, val_ds, test_ds = DataSplitter.split(
        dataset,
        train_ratio=1 - val_ratio - test_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=42
    )

    # Initialize featurizer
    featurizer = MolecularFeaturizer()

    # Convert a split to graph dicts + targets
    def featurize(split):
        # Convert all SMILES → graph objects
        graphs = featurizer.batch_smiles_to_graphs(split.smiles)
        targets = jnp.array(split.targets, dtype=float)

        # Keep only valid graphs
        valid_idx = [i for i, g in enumerate(graphs) if g is not None]
        valid_idx_jax = jnp.array(valid_idx, dtype=int)

        graphs = [graphs[i] for i in valid_idx]
        targets = targets[valid_idx_jax]

        # Convert to dict format
        dict_graphs = [{
            "node_features": g.node_features,
            "edge_index": g.edge_index,
            "edge_features": g.edge_features
        } for g in graphs]

        return dict_graphs, targets

    print("   Training set...")
    train_graphs, train_targets = featurize(train_ds)

    print("   Validation set...")
    val_graphs, val_targets = featurize(val_ds)

    print("   Test set...")
    test_graphs, test_targets = featurize(test_ds)

    print(f"\n✅ Data prepared:")
    print(f"   Train: {len(train_graphs)}")
    print(f"   Val:   {len(val_graphs)}")
    print(f"   Test:  {len(test_graphs)}")

    return (
        train_graphs, train_targets,
        val_graphs, val_targets,
        test_graphs, test_targets,
        featurizer
    )


# ============================================================
# 🧠 STEP 2 — MODEL INITIALIZATION
# ============================================================
def create_model(featurizer, hidden_dim=128, num_layers=3, lr=1e-3):

    print("=" * 70)
    print("🧠 STEP 2: MODEL INITIALIZATION")
    print("=" * 70)

    # Create model
    model = GNNPredictor(
        node_feat_dim=featurizer.node_feat_dim,
        edge_feat_dim=featurizer.edge_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
    )

    # Dummy graph → initialize params
    dummy = GNNPredictor.create_dummy_graph(
        featurizer.node_feat_dim, featurizer.edge_feat_dim
    )
    params = model.init_params(dummy)

    # Optimizer
    schedule = optax.cosine_decay_schedule(lr, decay_steps=20000)
    optimizer = optax.adam(schedule)

    return model, params, optimizer


# ============================================================
# 🏋️ STEP 3 — TRAINING
# ============================================================
def train_model(model, params, optimizer,
                train_data, val_data, epochs=60, patience=12):

    ckpt_dir = Path("experiments/checkpoints/classical_gnn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = GNNTrainer(
        model_forward_fn=model.forward,
        optimizer=optimizer,
        checkpoint_dir=str(ckpt_dir),
        seed=42
    )

    train_graphs, train_targets = train_data
    val_graphs, val_targets = val_data

    best_params, _, metrics = trainer.fit(
        params=params,
        train_graphs=train_graphs,
        train_targets=train_targets,
        val_graphs=val_graphs,
        val_targets=val_targets,
        num_epochs=epochs,
        patience=patience,
        verbose=True
    )

    return best_params


# ============================================================
# 🎯 MAIN ENTRYPOINT
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="mu")
    parser.add_argument("--max_samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    (train_graphs, train_targets,
    val_graphs, val_targets,
    test_graphs, test_targets,
    featurizer) = prepare_data(
        target_property=args.target,
        max_samples=args.max_samples
    )

    model, params, optimizer = create_model(featurizer)

    best_params = train_model(
        model,
        params,
        optimizer,
        (train_graphs, train_targets),
        (val_graphs, val_targets),
        epochs=args.epochs
    )

    print("\n🎯 DONE: Classical GNN trained successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
    