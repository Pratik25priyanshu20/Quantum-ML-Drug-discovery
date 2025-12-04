"""
Quantum Data Encoding
Convert classical molecular features to quantum states
"""

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import pickle
from pathlib import Path


class QuantumEncoder:
    """
    Encode classical molecular features into quantum states
    
    Strategy:
    1. High-dimensional classical features (45D from graph)
    2. Dimensionality reduction (PCA to n_qubits dimensions)
    3. Normalization to [0, π] for angle encoding
    """
    
    def __init__(self, n_qubits: int = 4):
        """
        Args:
            n_qubits: Number of qubits (determines encoding dimension)
        """
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=n_qubits)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        print(f"✅ Quantum Encoder initialized:")
        print(f"   Qubits: {n_qubits}")
        print(f"   Output dimension: {n_qubits}")
        print(f"   Encoding range: [0, π]")
    
    def fit(self, features: np.ndarray):
        """
        Fit PCA and scaler on training data
        
        Args:
            features: [num_samples, feature_dim] classical features
        """
        print(f"\n🔧 Fitting quantum encoder on {len(features)} samples...")
        
        # Fit scaler
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Fit PCA
        self.pca.fit(features_scaled)
        
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"   PCA explained variance: {cumulative_var[-1]:.2%}")
        print(f"   Per component: {explained_var}")
        
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> jnp.ndarray:
        """
        Transform classical features to quantum encoding
        
        Args:
            features: [num_samples, feature_dim] or [feature_dim]
            
        Returns:
            Quantum encoding [num_samples, n_qubits] or [n_qubits]
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        single_sample = features.ndim == 1
        if single_sample:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply PCA
        features_reduced = self.pca.transform(features_scaled)
        
        # Normalize to [0, π] for angle encoding
        # Strategy: min-max normalization per feature
        features_normalized = self._normalize_to_pi(features_reduced)
        
        if single_sample:
            features_normalized = features_normalized.squeeze(0)
        
        return jnp.array(features_normalized, dtype=jnp.float32)
    
    def _normalize_to_pi(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, π] range
        
        Uses min-max normalization with robust bounds
        """
        # Use percentiles for robust normalization
        lower = np.percentile(features, 1, axis=0)
        upper = np.percentile(features, 99, axis=0)
        
        # Avoid division by zero
        range_vals = upper - lower
        range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)
        
        # Normalize to [0, 1]
        normalized = (features - lower) / range_vals
        normalized = np.clip(normalized, 0, 1)
        
        # Scale to [0, π]
        return normalized * np.pi
    
    def fit_transform(self, features: np.ndarray) -> jnp.ndarray:
        """Fit and transform in one step"""
        self.fit(features)
        return self.transform(features)
    
    def save(self, path: str):
        """Save encoder to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_qubits': self.n_qubits,
                'pca': self.pca,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)
        print(f"💾 Encoder saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load encoder from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        encoder = cls(n_qubits=data['n_qubits'])
        encoder.pca = data['pca']
        encoder.scaler = data['scaler']
        encoder.is_fitted = data['is_fitted']
        
        print(f"📂 Encoder loaded from {path}")
        return encoder


class GraphToVectorFeatures:
    """
    Convert molecular graphs to fixed-size feature vectors
    for quantum encoding
    """
    
    @staticmethod
    def aggregate_node_features(graph_dict: dict) -> np.ndarray:
        """
        Aggregate node features from graph
        
        Strategies:
        - Mean pooling
        - Max pooling
        - Sum pooling
        - Standard deviation
        
        Args:
            graph_dict: Dictionary with 'node_features', 'edge_index', 'edge_features'
            
        Returns:
            Fixed-size feature vector
        """
        node_features = np.array(graph_dict['node_features'])
        
        # Multiple aggregations
        features = []
        
        # Mean
        features.append(np.mean(node_features, axis=0))
        
        # Max
        features.append(np.max(node_features, axis=0))
        
        # Min
        features.append(np.min(node_features, axis=0))
        
        # Std
        features.append(np.std(node_features, axis=0))
        
        # Concatenate all
        aggregated = np.concatenate(features)
        
        return aggregated
    
    @staticmethod
    def batch_aggregate(graph_list: List[dict]) -> np.ndarray:
        """
        Aggregate features for a batch of graphs
        
        Returns:
            [num_graphs, feature_dim]
        """
        features = []
        for graph in graph_list:
            feat = GraphToVectorFeatures.aggregate_node_features(graph)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)


def prepare_quantum_dataset(
    graphs: List[dict],
    targets: jnp.ndarray,
    encoder: Optional[QuantumEncoder] = None,
    n_qubits: int = 4,
    fit_encoder: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, QuantumEncoder]:
    """
    Prepare dataset for quantum training
    
    Args:
        graphs: List of graph dictionaries
        targets: Target values
        encoder: Pre-fitted encoder (optional)
        n_qubits: Number of qubits if creating new encoder
        fit_encoder: Whether to fit encoder on this data
        
    Returns:
        (quantum_features, targets, encoder)
    """
    print(f"\n🔄 Preparing quantum dataset...")
    print(f"   Graphs: {len(graphs)}")
    print(f"   Target qubits: {n_qubits}")
    
    # Convert graphs to vectors
    print("   Aggregating graph features...")
    classical_features = GraphToVectorFeatures.batch_aggregate(graphs)
    print(f"   Classical features shape: {classical_features.shape}")
    
    # Create or use encoder
    if encoder is None:
        encoder = QuantumEncoder(n_qubits=n_qubits)
        fit_encoder = True
    
    # Encode features
    if fit_encoder:
        quantum_features = encoder.fit_transform(classical_features)
        print(f"   ✅ Encoder fitted")
    else:
        quantum_features = encoder.transform(classical_features)
    
    print(f"   Quantum features shape: {quantum_features.shape}")
    print(f"   Quantum features range: [{quantum_features.min():.3f}, {quantum_features.max():.3f}]")
    
    return quantum_features, targets, encoder


def test_quantum_encoding():
    """Test quantum encoding pipeline"""
    print("🧪 Testing Quantum Encoding\n")
    
    # Create dummy molecular graphs
    np.random.seed(42)
    dummy_graphs = [
        {
            'node_features': np.random.randn(np.random.randint(5, 15), 45),
            'edge_index': np.array([[0, 1], [1, 0]]),
            'edge_features': np.random.randn(2, 6)
        }
        for _ in range(100)
    ]
    dummy_targets = jnp.array(np.random.randn(100))
    
    # Test graph to vector
    print("1️⃣  Testing Graph → Vector:")
    features = GraphToVectorFeatures.batch_aggregate(dummy_graphs[:5])
    print(f"   Input: {len(dummy_graphs[:5])} graphs")
    print(f"   Output: {features.shape}")
    print(f"   Sample feature range: [{features[0].min():.3f}, {features[0].max():.3f}]")
    
    # Test quantum encoder
    print("\n2️⃣  Testing Quantum Encoder:")
    encoder = QuantumEncoder(n_qubits=4)
    
    # Fit on training data
    train_features = GraphToVectorFeatures.batch_aggregate(dummy_graphs[:80])
    encoder.fit(train_features)
    
    # Transform
    quantum_train = encoder.transform(train_features)
    print(f"   Train features: {train_features.shape} → {quantum_train.shape}")
    print(f"   Quantum range: [{quantum_train.min():.3f}, {quantum_train.max():.3f}]")
    
    # Transform test (unseen data)
    test_features = GraphToVectorFeatures.batch_aggregate(dummy_graphs[80:])
    quantum_test = encoder.transform(test_features)
    print(f"   Test features: {test_features.shape} → {quantum_test.shape}")
    
    # Test single sample
    print("\n3️⃣  Testing Single Sample:")
    single_feat = GraphToVectorFeatures.aggregate_node_features(dummy_graphs[0])
    quantum_single = encoder.transform(single_feat)
    print(f"   Single input: {single_feat.shape} → {quantum_single.shape}")
    print(f"   Quantum values: {quantum_single}")
    
    # Test save/load
    print("\n4️⃣  Testing Save/Load:")
    encoder.save("data/processed/test_encoder.pkl")
    encoder_loaded = QuantumEncoder.load("data/processed/test_encoder.pkl")
    quantum_loaded = encoder_loaded.transform(single_feat)
    assert np.allclose(quantum_single, quantum_loaded)
    print("   ✅ Save/load successful")
    
    # Test full pipeline
    print("\n5️⃣  Testing Full Pipeline:")
    q_features, q_targets, q_encoder = prepare_quantum_dataset(
        graphs=dummy_graphs,
        targets=dummy_targets,
        n_qubits=4,
        fit_encoder=True
    )
    print(f"   ✅ Pipeline complete: {q_features.shape}")
    
    print("\n✅ All quantum encoding tests passed!")


if __name__ == "__main__":
    test_quantum_encoding()