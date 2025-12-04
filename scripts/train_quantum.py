"""
Train Quantum Model
Full quantum training pipeline
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
import numpy as np

from src.data.loaders import QM9Loader, DataSplitter
from src.data.molecular_features import MolecularFeaturizer
from src.data.quantum_encoding import prepare_quantum_dataset, QuantumEncoder
from src.models.quantum.quantum_circuits import QuantumNeuralNetwork
from src.training.quantum_trainer import QuantumTrainer


def prepare_data(target_property='mu', max_samples=10000, n_qubits=4):
    """
    Load and prepare data for quantum training
    
    Returns:
        Quantum-encoded train/val/test data + encoder
    """
    print("=" * 70)
    print("📦 STEP 1: DATA PREPARATION")
    print("=" * 70)
    
    # Load QM9
    loader = QM9Loader()
    dataset = loader.load(target_property=target_property, max_samples=max_samples)
    
    # Split data
    train_ds, val_ds, test_ds = DataSplitter.split(
        dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42
    )
    
    # Featurize molecules to graphs
    print(f"\n⚗️  Featurizing molecules...")
    featurizer = MolecularFeaturizer()
    
    print("   Training set...")
    train_graphs = featurizer.batch_smiles_to_graphs(train_ds.smiles, show_progress=True)
    
    print("   Validation set...")
    val_graphs = featurizer.batch_smiles_to_graphs(val_ds.smiles, show_progress=True)
    
    print("   Test set...")
    test_graphs = featurizer.batch_smiles_to_graphs(test_ds.smiles, show_progress=True)
    
    # Filter valid
    def filter_valid(graphs, targets):
        valid_idx = [i for i, g in enumerate(graphs) if g is not None]
        targets_np = np.array(targets)
        return [graphs[i] for i in valid_idx], targets_np[valid_idx]
    train_graphs, train_targets = filter_valid(train_graphs, train_ds.targets)
    val_graphs, val_targets = filter_valid(val_graphs, val_ds.targets)
    test_graphs, test_targets = filter_valid(test_graphs, test_ds.targets)
    
    # Convert to dicts
    def to_dicts(graph_list):
        return [{'node_features': g.node_features, 'edge_index': g.edge_index,
                 'edge_features': g.edge_features} for g in graph_list]
    
    train_graphs = to_dicts(train_graphs)
    val_graphs = to_dicts(val_graphs)
    test_graphs = to_dicts(test_graphs)
    
    print(f"\n✅ Graphs prepared:")
    print(f"   Train: {len(train_graphs)}")
    print(f"   Val:   {len(val_graphs)}")
    print(f"   Test:  {len(test_graphs)}")
    
    # Quantum encoding
    print(f"\n⚛️  Quantum encoding...")
    train_q_features, _, encoder = prepare_quantum_dataset(
        train_graphs, train_targets, n_qubits=n_qubits, fit_encoder=True
    )
    
    val_q_features, _, _ = prepare_quantum_dataset(
        val_graphs, val_targets, encoder=encoder, fit_encoder=False
    )
    
    test_q_features, _, _ = prepare_quantum_dataset(
        test_graphs, test_targets, encoder=encoder, fit_encoder=False
    )
    
    # Save encoder
    encoder.save("data/processed/quantum_encoder.pkl")
    
    print(f"\n✅ Quantum data prepared:")
    print(f"   Train: {train_q_features.shape}")
    print(f"   Val:   {val_q_features.shape}")
    print(f"   Test:  {test_q_features.shape}")
    
    return (train_q_features, train_targets, val_q_features, val_targets,
            test_q_features, test_targets, encoder)


def create_model(n_qubits=4, n_layers=2, learning_rate=0.01):
    """
    Create quantum model
    """
    print("\n" + "=" * 70)
    print("⚛️  STEP 2: QUANTUM MODEL INITIALIZATION")
    print("=" * 70)
    
    # Create QNN
    qnn = QuantumNeuralNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        output_dim=1,
        feature_map='angle',
        entanglement='linear',
        seed=42
    )
    
    # Initialize parameters
    params = qnn.initialize_parameters()
    
    # Optimizer: clip grads + cosine decay (gentler than fixed LR)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=10000
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule)
    )
    
    print(f"\n✅ Quantum model created:")
    print(f"   Total parameters: {qnn.count_parameters()}")
    print(f"   Learning rate: {learning_rate}")
    
    return qnn, params, optimizer


def train_model(qnn, params, optimizer, train_data, val_data, 
                num_epochs=50, patience=10, run_name="quantum_vqc"):
    """
    Train quantum model
    """
    print("\n" + "=" * 70)
    print("🏋️  STEP 3: QUANTUM TRAINING")
    print("=" * 70)
    
    train_features, train_targets, val_features, val_targets = train_data + val_data
    
    # Create trainer
    trainer = QuantumTrainer(
        forward_fn=qnn.forward,
        optimizer=optimizer,
        checkpoint_dir=f"experiments/checkpoints/{run_name}",
        seed=42
    )
    
    # Train
    start_time = time.time()
    
    best_params, opt_state, history = trainer.fit(
        params=params,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        num_epochs=num_epochs,
        patience=patience,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️  Training time: {training_time/60:.2f} minutes")
    
    return best_params, history


def evaluate_model(qnn, params, test_features, test_targets, run_name="quantum_vqc"):
    """
    Evaluate quantum model
    """
    print("\n" + "=" * 70)
    print("📊 STEP 4: TEST EVALUATION")
    print("=" * 70)
    
    # Make predictions
    predictions = []
    for features in test_features:
        pred = qnn.forward(params, features)
        predictions.append(float(pred))
    
    predictions = jnp.array(predictions)
    
    # Calculate metrics
    mae = jnp.mean(jnp.abs(predictions - test_targets))
    rmse = jnp.sqrt(jnp.mean((predictions - test_targets) ** 2))
    
    ss_res = jnp.sum((test_targets - predictions) ** 2)
    ss_tot = jnp.sum((test_targets - jnp.mean(test_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    corr = jnp.corrcoef(predictions, test_targets)[0, 1]
    
    print(f"\n📈 Test Set Performance:")
    print(f"   MAE:  {float(mae):.4f} Debye")
    print(f"   RMSE: {float(rmse):.4f} Debye")
    print(f"   R²:   {float(r2):.4f}")
    print(f"   Corr: {float(corr):.4f}")
    
    # Save results
    results = {
        'predictions': [float(p) for p in predictions],
        'targets': [float(t) for t in test_targets],
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'correlation': float(corr)
        }
    }
    
    results_path = Path("experiments/results") / f"{run_name}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Quantum VQC')
    parser.add_argument('--target', type=str, default='mu', help='Target property')
    parser.add_argument('--max_samples', type=int, default=5000, 
                       help='Max samples (quantum is slow!)')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--n_layers', type=int, default=2, help='VQC layers')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--run_name', type=str, default='quantum_vqc',
                       help='Name for checkpoints/results folders')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("⚛️  QUANTUM DRUG DISCOVERY - VARIATIONAL QUANTUM CLASSIFIER")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   Target: {args.target}")
    print(f"   Max samples: {args.max_samples}")
    print(f"   Qubits: {args.n_qubits}")
    print(f"   VQC layers: {args.n_layers}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Max epochs: {args.epochs}")
    print(f"\n⚠️  Note: Quantum training is SLOW (~1 sample/sec)")
    print(f"   Estimated time: ~{args.max_samples * 0.8 * args.epochs / 3600:.1f} hours")
    print()
    
    # Prepare data
    train_features, train_targets, val_features, val_targets, \
    test_features, test_targets, encoder = prepare_data(
        target_property=args.target,
        max_samples=args.max_samples,
        n_qubits=args.n_qubits
    )
    
    # Create model
    qnn, params, optimizer = create_model(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        learning_rate=args.lr
    )
    
    # Train
    best_params, history = train_model(
        qnn, params, optimizer,
        (train_features, train_targets),
        (val_features, val_targets),
        num_epochs=args.epochs,
        patience=args.patience,
        run_name=args.run_name
    )
    
    # Evaluate
    results = evaluate_model(qnn, best_params, test_features, test_targets,
                             run_name=args.run_name)
    
    print("\n" + "=" * 70)
    print("✅ QUANTUM TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n🎯 Final Test MAE: {results['metrics']['mae']:.4f} Debye")
    print(f"⚛️  Checkpoints: experiments/checkpoints/{args.run_name}/")
    print(f"📈 Results: experiments/results/{args.run_name}_results.json")
    


if __name__ == "__main__":
    main()
