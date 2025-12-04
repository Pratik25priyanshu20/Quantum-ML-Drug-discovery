"""
Train Hybrid Quantum-Classical Model
FULLY FIXED version – compatible with global QNode + Option-A hybrid model.
"""

import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import time
import argparse
import json

from src.data.loaders import QM9Loader, DataSplitter
from src.data.molecular_features import MolecularFeaturizer
from src.models.hybrid.hybrid_model import HybridRegressor
from src.training.hybrid_trainer import HybridTrainer


# ================================================================
#  DATA PREPARATION
# ================================================================
def prepare_data(target_property='mu', max_samples=5000):

    print("=" * 70)
    print("📦 STEP 1: DATA PREPARATION")
    print("=" * 70)

    loader = QM9Loader()
    dataset = loader.load(target_property=target_property, max_samples=max_samples)

    train_ds, val_ds, test_ds = DataSplitter.split(
        dataset, 0.8, 0.1, 0.1, random_seed=42
    )

    # Featurize
    print("\n⚗️  Featurizing molecules...")
    featurizer = MolecularFeaturizer()

    print("   Training set...")
    train_graphs = featurizer.batch_smiles_to_graphs(train_ds.smiles, show_progress=True)

    print("   Validation set...")
    val_graphs = featurizer.batch_smiles_to_graphs(val_ds.smiles, show_progress=True)

    print("   Test set...")
    test_graphs = featurizer.batch_smiles_to_graphs(test_ds.smiles, show_progress=True)

    # FIXED filter_valid (no JAX array indexing)
    def filter_valid(graphs, targets):
        valid_idx = [i for i, g in enumerate(graphs) if g is not None]
        filtered_graphs = [graphs[i] for i in valid_idx]
        filtered_targets = jnp.array([targets[i] for i in valid_idx])
        return filtered_graphs, filtered_targets

    train_graphs, train_targets = filter_valid(train_graphs, train_ds.targets)
    val_graphs, val_targets = filter_valid(val_graphs, val_ds.targets)
    test_graphs, test_targets = filter_valid(test_graphs, test_ds.targets)

    # Convert to dictionary format expected by hybrid model
    def to_dicts(graphs):
        return [{
            "node_features": g.node_features,
            "edge_index": g.edge_index,
            "edge_features": g.edge_features
        } for g in graphs]

    train_graphs = to_dicts(train_graphs)
    val_graphs = to_dicts(val_graphs)
    test_graphs = to_dicts(test_graphs)

    print("\n✅ Data prepared:")
    print(f"   Train: {len(train_graphs)}")
    print(f"   Val:   {len(val_graphs)}")
    print(f"   Test:  {len(test_graphs)}")

    return (
        train_graphs, train_targets,
        val_graphs, val_targets,
        test_graphs, test_targets,
        featurizer
    )


# ================================================================
#  MODEL CREATION
# ================================================================
def create_model(featurizer, gnn_hidden_dim=128, gnn_layers=2,
                 n_qubits=4, quantum_layers=2, decoder_dims=(64, 32)):

    print("\n" + "=" * 70)
    print("🔮 STEP 2: HYBRID MODEL INITIALIZATION")
    print("=" * 70)

    predictor = HybridRegressor(
        node_feat_dim=featurizer.node_feat_dim,
        edge_feat_dim=featurizer.edge_feat_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        n_qubits=n_qubits,
        quantum_layers=quantum_layers,
        decoder_hidden_dims=decoder_dims,
        output_dim=1,
        seed=42
    )

    dummy_graph = HybridRegressor.create_dummy_graph(
        featurizer.node_feat_dim, featurizer.edge_feat_dim
    )
    params = predictor.init_params(dummy_graph)

    # Stage-1 optimizer: classical components
    optimizer_stage1 = optax.adam(1e-3)

    # Stage-2 optimizer: fine-tune quantum + classical
    optimizer_stage2 = optax.adam(5e-4)

    print("\n✅ Hybrid model created")
    print(f"   Total parameters: {predictor.count_parameters():,}")
    print(f"   Stage 1 LR: 1e-3")
    print(f"   Stage 2 LR: 5e-4")

    return predictor, params, optimizer_stage1, optimizer_stage2


# ================================================================
#  TRAINING PIPELINE
# ================================================================
def train_model(
    predictor,
    params,
    optimizer_stage1,
    optimizer_stage2,
    train_data,
    val_data,
    stage1_epochs=30,
    stage2_epochs=20,
    patience=15
):

    print("\n" + "=" * 70)
    print("🏋️  STEP 3: TWO-STAGE TRAINING")
    print("=" * 70)

    train_graphs, train_targets = train_data
    val_graphs, val_targets = val_data

    trainer = HybridTrainer(
        model_forward_fn=predictor.forward,
        optimizer_classical=optimizer_stage1,
        optimizer_full=optimizer_stage2,
        checkpoint_dir="experiments/checkpoints/hybrid_model",
        seed=42
    )

    start = time.time()

    best_params, history = trainer.fit(
        params,
        train_graphs,
        train_targets,
        val_graphs,
        val_targets,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        patience=patience,
        verbose=True
    )

    print(f"\n⏱ Training time: {(time.time() - start)/60:.2f} minutes")

    return best_params, history


# ================================================================
#  EVALUATION
# ================================================================
def evaluate_model(predictor, params, test_graphs, test_targets):

    print("\n" + "=" * 70)
    print("📊 STEP 4: TEST EVALUATION")
    print("=" * 70)

    preds = []
    for g in test_graphs:
        preds.append(float(predictor.predict(params, g)))

    preds = jnp.array(preds)

    mae = jnp.mean(jnp.abs(preds - test_targets))
    rmse = jnp.sqrt(jnp.mean((preds - test_targets) ** 2))

    ss_res = jnp.sum((test_targets - preds) ** 2)
    ss_tot = jnp.sum((test_targets - jnp.mean(test_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    corr = jnp.corrcoef(preds, test_targets)[0, 1]

    print("\n📈 Test Results:")
    print(f"   MAE:  {float(mae):.4f}")
    print(f"   RMSE: {float(rmse):.4f}")
    print(f"   R²:   {float(r2):.4f}")
    print(f"   Corr: {float(corr):.4f}")

    return {
        "predictions": [float(x) for x in preds],
        "targets": [float(x) for x in test_targets],
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "corr": float(corr),
        }
    }


# ================================================================
#  MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="mu")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--quantum_layers", type=int, default=2)
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔮 HYBRID QUANTUM–CLASSICAL TRAINING")
    print("=" * 70)

    (
        train_graphs, train_targets,
        val_graphs, val_targets,
        test_graphs, test_targets,
        featurizer
    ) = prepare_data(args.target, args.max_samples)

    predictor, params, opt1, opt2 = create_model(
        featurizer,
        gnn_hidden_dim=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        n_qubits=args.n_qubits,
        quantum_layers=args.quantum_layers
    )

    best_params, history = train_model(
        predictor, params, opt1, opt2,
        (train_graphs, train_targets),
        (val_graphs, val_targets),
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        patience=args.patience
    )

    results = evaluate_model(predictor, best_params, test_graphs, test_targets)

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
