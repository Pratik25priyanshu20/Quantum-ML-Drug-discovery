# api/app.py

import uvicorn
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import traceback
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

# --------------------------
# PROJECT IMPORTS
# --------------------------
from src.data.molecular_features import MolecularFeaturizer
from src.models.classical.gnn_baseline import GNNPredictor
from src.models.hybrid.hybrid_model import HybridRegressor
from src.models.quantum.quantum_circuits import QuantumNeuralNetwork
from src.data.quantum_encoding import QuantumEncoder, GraphToVectorFeatures


# ============================================================
# FASTAPI SETUP
# ============================================================
app = FastAPI(
    title="Quantum Drug Discovery API",
    description="Predict molecular properties using Classical and Hybrid ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

featurizer = MolecularFeaturizer()


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================
class PredictRequest(BaseModel):
    smiles: str
    models: List[str] = ["classical"]


class PredictionResponse(BaseModel):
    valid: bool
    smiles: str
    predictions: Dict[str, float] = {}
    inference_times: Dict[str, float] = {}
    molecular_properties: Dict[str, float] = {}
    error: str = None


# ============================================================
# MOLECULAR PROPERTIES (RDKit)
# ============================================================
def get_molecular_properties(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "tpsa": float(Descriptors.TPSA(mol)),
        }
    except:
        return {}


# ============================================================
# SMILES → GRAPH
# ============================================================
def smiles_to_graph(smiles: str):
    try:
        g = featurizer.smiles_to_graph(smiles)
        if g is None:
            return None

        return {
            "node_features": jnp.array(g.node_features),
            "edge_index": jnp.array(g.edge_index),
            "edge_features": jnp.array(g.edge_features),
        }
    except Exception as e:
        print("❌ SMILES conversion error:", e)
        return None


# ============================================================
# CHECKPOINT LOADER (NEW FORMAT)
# ============================================================
def load_checkpoint_flat(model, ckpt_path: Path):
    """
    Load the new param_0, param_1, ..., param_N flattened checkpoint.
    Rebuilds the correct Flax param tree using the model's init params.
    """
    if not ckpt_path.exists():
        print("❌ Checkpoint missing:", ckpt_path)
        return None

    print("📦 Loading checkpoint:", ckpt_path)

    ckpt = np.load(ckpt_path, allow_pickle=True)

    # 1) Get param_*
    param_keys = [k for k in ckpt.files if k.startswith("param_")]
    param_keys = sorted(param_keys, key=lambda x: int(x.split("_")[1]))

    flat_arrays = [ckpt[k] for k in param_keys]

    # 2) Create dummy graph to obtain correct treedef
    dummy = model.create_dummy_graph(
        featurizer.node_feat_dim,
        featurizer.edge_feat_dim
    )

    init_params = model.init_params(dummy)
    tree_def = jax.tree_util.tree_structure(init_params)

    # 3) Unflatten into correct flax structure
    params = jax.tree_util.tree_unflatten(tree_def, flat_arrays)

    print(f"✅ Loaded {len(flat_arrays)} tensors")
    return params


# ============================================================
# LOAD MODELS
# ============================================================
def load_classical_model():
    try:
        model = GNNPredictor(
            node_feat_dim=featurizer.node_feat_dim,
            edge_feat_dim=featurizer.edge_feat_dim,
            hidden_dim=128,
            num_layers=3,
            output_dim=1,
        )

        ckpt_path = Path("experiments/checkpoints/classical_gnn/best.npz")
        params = load_checkpoint_flat(model, ckpt_path)
        return model, params

    except Exception as e:
        print("❌ Classical load error:", e)
        traceback.print_exc()
        return None, None


def load_hybrid_model():
    try:
        model = HybridRegressor(
            node_feat_dim=featurizer.node_feat_dim,
            edge_feat_dim=featurizer.edge_feat_dim,
            gnn_hidden_dim=128,
            gnn_layers=2,
            n_qubits=4,
            quantum_layers=2,
            decoder_hidden_dims=(64, 32),
            output_dim=1
        )

        ckpt_path = Path("experiments/checkpoints/hybrid_model/stage2_best.npz")
        params = load_checkpoint_flat(model, ckpt_path)
        return model, params

    except Exception as e:
        print("❌ Hybrid load error:", e)
        traceback.print_exc()
        return None, None


def load_quantum_model():
    """
    Load quantum checkpoint + encoder for inference.
    """
    try:
        encoder_path = Path("data/processed/quantum_encoder.pkl")
        if not encoder_path.exists():
            print("❌ Quantum encoder missing:", encoder_path)
            return None, None, None
        encoder = QuantumEncoder.load(str(encoder_path))

        ckpt_path = Path("experiments/checkpoints/quantum_vqc_full/best.npz")
        if not ckpt_path.exists():
            print("❌ Quantum checkpoint missing:", ckpt_path)
            return None, None, None

        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, output_dim=1)
        dummy = qnn.initialize_parameters()
        leaves, treedef = jax.tree_util.tree_flatten(dummy)
        ckpt = np.load(ckpt_path, allow_pickle=True)
        params = jax.tree_util.tree_unflatten(
            treedef,
            [jnp.array(ckpt[f"param_{i}"]) for i in range(len(leaves))]
        )
        return qnn, params, encoder

    except Exception as e:
        print("❌ Quantum load error:", e)
        traceback.print_exc()
        return None, None, None


print("\n🚀 Loading models...")
classical_model, classical_params = load_classical_model()
hybrid_model, hybrid_params = load_hybrid_model()
quantum_model, quantum_params, quantum_encoder = load_quantum_model()

models_loaded = {
    "classical": classical_params is not None,
    "hybrid": hybrid_params is not None,
    "quantum": quantum_params is not None,
}

print("📌 Models loaded:", models_loaded)


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def health():
    return {
        "status": "ok",
        "models_loaded": models_loaded
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    try:
        graph = smiles_to_graph(request.smiles)
        if graph is None:
            return PredictionResponse(
                valid=False,
                smiles=request.smiles,
                error="Invalid SMILES"
            )

        predictions = {}
        inference_times = {}

        # CLASSICAL MODEL
        if "classical" in request.models and classical_params is not None:
            t0 = time.time()
            p = classical_model.predict(classical_params, graph)
            predictions["classical"] = float(p)
            inference_times["classical"] = time.time() - t0

        # HYBRID MODEL
        if "hybrid" in request.models and hybrid_params is not None:
            t0 = time.time()
            p = hybrid_model.predict(hybrid_params, graph)
            predictions["hybrid"] = float(p)
            inference_times["hybrid"] = time.time() - t0

        # QUANTUM MODEL
        if "quantum" in request.models and quantum_params is not None:
            try:
                # Aggregate graph -> vector -> quantum encode
                vec = GraphToVectorFeatures.aggregate_node_features({
                    "node_features": np.array(graph["node_features"]),
                    "edge_index": np.array(graph["edge_index"]),
                    "edge_features": np.array(graph["edge_features"])
                })
                qx = quantum_encoder.transform(vec)
                t0 = time.time()
                p = quantum_model.forward(quantum_params, jnp.array(qx))
                predictions["quantum"] = float(p)
                inference_times["quantum"] = time.time() - t0
            except Exception as qe:
                print("❌ Quantum inference error:", qe)

        props = get_molecular_properties(request.smiles)

        return PredictionResponse(
            valid=True,
            smiles=request.smiles,
            predictions=predictions,
            inference_times=inference_times,
            molecular_properties=props,
        )

    except Exception as e:
        print("❌ Prediction ERROR:", e)
        traceback.print_exc()
        return PredictionResponse(
            valid=False,
            smiles=request.smiles,
            error=str(e)
        )


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Starting API at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
