import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, rdMolDescriptors
from rdkit import DataStructs
import py3Dmol
import os
import json
from streamlit import components

# Page config
st.set_page_config(page_title="Quantum Drug Discovery", layout="wide")

# Optional mordred descriptors
try:
    from mordred import Calculator, descriptors
    desc_calc = Calculator(descriptors, ignore_3D=False)
except Exception:
    desc_calc = None

# ----------------------------
# Theme CSS (dark + bright) and helpers
# ----------------------------
DARK_CSS = """
<style>
.main {background-color: #0e1117 !important;}
.block-container {padding-top: 1.5rem;}
* {font-family: 'Inter', sans-serif;}
@keyframes fadeIn {from {opacity: 0; transform: translateY(-10px);} to {opacity: 1; transform: translateY(0);}}
.success-banner {background-color: rgba(46, 204, 113, 0.2); padding: 12px 18px;
border-left: 4px solid #2ecc71; border-radius: 6px; animation: fadeIn 0.6s ease-in-out;
color: #2ecc71; font-weight: 600;}
.section-title {font-size: 1.4rem; font-weight: 700; color: #cfcfcf; margin-top: 1rem;}
.glass-card {background: rgba(255, 255, 255, 0.08); box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 18px; padding: 16px; border: 1px solid rgba(255, 255, 255, 0.12);}
.hero {padding: 14px 16px; border-radius: 14px; background: radial-gradient(circle at top, #1f2a3d 0%, #0f172a 100%); color: #e5e7eb; box-shadow: 0 12px 30px rgba(0,0,0,0.35);}
.metric-card {padding: 12px; border-radius: 12px; background: linear-gradient(135deg,#1c2433 0%,#101725 100%); color: #e5e7eb; box-shadow: 0 10px 26px rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06);}
.viewer-box {background: rgba(255, 255, 255, 0.09); border-radius: 15px; padding: 10px; border: 1px solid rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px);}
.chart-card {background: rgba(255, 255, 255, 0.1); padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.12);}
.floating-btn {position: fixed; bottom: 35px; right: 35px; background: linear-gradient(135deg, #00ffc6, #7d2cff); padding: 14px 28px; border-radius: 12px; color: black !important; font-size: 18px; font-weight: 700; box-shadow: 0 4px 25px rgba(0,255,200,0.4); cursor: pointer; z-index: 999; transition: all 0.3s ease;}
.floating-btn:hover {transform: scale(1.05); box-shadow: 0 6px 30px rgba(150,0,255,0.5);}
.shimmer {background: linear-gradient(90deg, rgba(255,255,255,0.06) 25%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0.06) 75%); background-size: 300% 100%; animation: shimmer 2s infinite; border-radius: 12px; height: 8px;}
div[data-testid="stNotification"] {display: none;}
.mol-bg {position: fixed; top: 10%; left: 5%; opacity: 0.08; animation: float-mol 6s infinite ease-in-out; z-index: -1; width: 300px;}
@keyframes float-mol {0% { transform: translateY(0px) rotate(0deg);}50% { transform: translateY(-25px) rotate(10deg);}100% { transform: translateY(0px) rotate(0deg);}}
</style>
"""

BRIGHT_CSS = """
<style>
body { background-color: #f5f7fb; }
.main { background-color: #f5f7fb !important; }
.block-container {padding-top: 1.5rem;}
.card { background: rgba(255,255,255,0.75); border-radius: 18px; padding: 18px; border: 1px solid rgba(180,180,180,0.25); box-shadow: 0 4px 18px rgba(0,0,0,0.08); transition: all 0.25s ease; }
.card:hover { box-shadow: 0 6px 26px rgba(0,0,0,0.15); transform: translateY(-3px);}
.section-title { font-size: 1.4rem; color: #1c273c; font-weight: 700; }
.metric-card { background: #ffffff; border-radius: 12px; padding: 14px; border: 1px solid rgba(170,170,170,0.25); box-shadow: 0 2px 14px rgba(0,0,0,0.06); }
.viewer-box { background: rgba(255,255,255,0.70); border-radius: 16px; padding: 10px; border: 1px solid rgba(180,180,180,0.25); }
.floating-btn {position: fixed; bottom: 35px; right: 35px; background: linear-gradient(135deg, #00ffc6, #7d2cff); padding: 14px 28px; border-radius: 12px; color: black !important; font-size: 18px; font-weight: 700; box-shadow: 0 4px 25px rgba(0,255,200,0.4); cursor: pointer; z-index: 999; transition: all 0.3s ease;}
.floating-btn:hover {transform: scale(1.05); box-shadow: 0 6px 30px rgba(150,0,255,0.5);}
.shimmer {background: linear-gradient(90deg, rgba(0,0,0,0.04) 25%, rgba(0,0,0,0.08) 50%, rgba(0,0,0,0.04) 75%); background-size: 300% 100%; animation: shimmer 2s infinite; border-radius: 12px; height: 8px;}
div[data-testid="stNotification"] {display: none;}
</style>
"""

theme = st.sidebar.selectbox("Theme", ["Dark", "Bright"], index=0)
st.markdown(DARK_CSS if theme == "Dark" else BRIGHT_CSS, unsafe_allow_html=True)
if theme == "Dark":
    st.markdown(
        """
        <img class="mol-bg" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Benzene_3D_ball.png/600px-Benzene_3D_ball.png">
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# API CONFIGURATION
# ----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ----------------------------
# Check API Health
# ----------------------------
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ----------------------------
# Request Prediction
# ----------------------------
def get_prediction(smiles, models):
    payload = {"smiles": smiles, "models": models}
    res = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
    return res.json()


# ----------------------------
# Draw molecule
# ----------------------------
def draw_molecule(smiles, size=(350, 350)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


# ----------------------------
# Prediction bar chart
# ----------------------------
def build_bar_chart(predictions):
    if not predictions:
        return None

    df = pd.DataFrame({
        "Model": list(predictions.keys()),
        "Value": list(predictions.values())
    })
    fig = px.bar(
        df,
        x="Model",
        y="Value",
        color="Model",
        text="Value",
        color_discrete_sequence=["#2E86AB", "#A23B72", "#F18F01"]
    )
    fig.update_traces(textposition="outside", hovertemplate="Model: %{x}<br>Value: %{y:.3f}<extra></extra>")
    fig.update_layout(
        title="Predicted Dipole Moment (Debye)",
        xaxis_title="Model",
        yaxis_title="Value",
        height=360,
        template="plotly_white",
        showlegend=False
    )
    return fig


# ----------------------------
# 3D Molecule renderer
# ----------------------------
def render_3d_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=320, height=320)
        viewer.addModel(block, "mol")
        viewer.setStyle({"stick": {"radius": 0.14, "colorscheme": "Jmol"}})
        viewer.setBackgroundColor("white")
        viewer.zoomTo()
        viewer.spin(True)
        return viewer
    except Exception:
        return None


def render_3d_rotating(xyz: str, speed=1):
    view = py3Dmol.view(width=500, height=400)
    view.addModel(xyz, "xyz")
    view.setStyle({"stick": {}})
    view.zoomTo()
    view.spin(True)
    view.setSpinParams(axis="y", rate=speed)
    return view


def render_viewer(viewer):
    if viewer is None:
        return None
    return components.v1.html(viewer._make_html(), height=430, width=480)


# ----------------------------
# Lipinski-style quick check
# ----------------------------
def lipinski_summary(props):
    try:
        mw = props.get("molecular_weight", 0)
        logp = props.get("logp", 0)
        rot = props.get("num_rotatable_bonds", 0)
        tpsa = props.get("tpsa", 0)
        rules = [
            ("MW ≤ 500", mw <= 500),
            ("LogP ≤ 5", logp <= 5),
            ("Rotatable bonds ≤ 10", rot <= 10),
            ("TPSA ≤ 140", tpsa <= 140),
        ]
        passed = sum(flag for _, flag in rules)
        lines = [f"{label}: {'✅' if flag else '⚠️'}" for label, flag in rules]
        return f"{passed}/4 rules passed\n" + "\n".join(lines)
    except Exception:
        return "N/A"


def compute_descriptors(mol):
    if desc_calc is None or mol is None:
        return {}
    try:
        vals = desc_calc(mol)
        as_dict = vals.asdict()
        return {k: float(v) for k, v in as_dict.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def generate_conformers(mol, n=3):
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
        energies = AllChem.MMFFOptimizeMoleculeConfs(mol)
        out = []
        for i, e in enumerate(energies):
            xyz = Chem.MolToXYZBlock(mol, confId=i)
            out.append((xyz, e[1]))
        return out
    except Exception:
        return []


st.markdown("""
<div class="hero">
  <h1>🧬 Quantum Drug Discovery</h1>
  <p>Predict molecular properties with Classical GNNs, Quantum VQC, and Hybrid models.</p>
  <hr/>
</div>
""", unsafe_allow_html=True)

if not check_api_health():
    st.error(f"⚠️ API not reachable at {API_URL}. Start with `docker compose up --build`.")
    st.stop()

tabs = st.tabs([
    "🔮 Predict",
    "🧬 Structures",
    "📊 Outputs",
    "💊 Drug-Likeness",
    "📊 Compare",
    "🧪 Examples",
    "🔍 Similarity",
    "ℹ️ About"
])
tab_predict, tab_struct, tab_outputs, tab_drug, tab_compare, tab_examples, tab_similarity, tab_about = tabs

# Floating predict button
st.markdown("""
<div class="floating-btn" onclick="const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Predict')); if (btn) { btn.click(); }">
    🚀 Predict
</div>
""", unsafe_allow_html=True)


# ===================================================================
# TAB 1 — PREDICT
# ===================================================================
with tab_predict:
    st.header("🔮 Molecular Property Prediction")
    col1, col2 = st.columns([1, 1])

    with col1:
        smiles = st.text_input("SMILES Input", "CCO")

        st.subheader("Select Models")
        classical = st.checkbox("Classical GNN", True)
        quantum = st.checkbox("Quantum VQC", False)
        hybrid = st.checkbox("Hybrid Q+GNN", True)

        model_list = [m for m, flag in [
            ("classical", classical),
            ("quantum", quantum),
            ("hybrid", hybrid)
        ] if flag]

        if st.button("🚀 Predict", use_container_width=True):
            if not model_list:
                st.warning("Select at least one model.")
            else:
                with st.spinner("Predicting..."):
                    st.session_state["pred"] = get_prediction(smiles, model_list)
                    st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)
                st.success("Prediction complete!")
                st.session_state["last_smiles"] = smiles

    with col2:
        data = st.session_state.get("pred")

        if data:
            if not data.get("valid", False):
                st.error(data.get("error", "Invalid SMILES."))
            else:
                st.markdown('<div class="success-banner">✅ Prediction Completed!</div>', unsafe_allow_html=True)
                fig = build_bar_chart(data["predictions"])
                if fig:
                    fig.update_layout(
                        height=360,
                        dragmode="pan",
                        modebar_add=["zoom", "zoomout", "autoscale"],
                        bargap=0.25
                    )
                    st.plotly_chart(fig, use_container_width=True, key="pred_chart_main")

                props = data.get("molecular_properties", {})
                # metric cards
                mw = props.get("molecular_weight", 0)
                logp = props.get("logp", 0)
                atoms = props.get("num_atoms", 0)
                dip = data["predictions"].get("classical", 0)
                rules_text = lipinski_summary(props)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Molecular Weight", f"{mw:.1f}")
                m2.metric("LogP", f"{logp:.2f}")
                m3.metric("# Atoms", atoms)
                m4.metric("Dipole (classical)", f"{dip:.2f}")
                m5.metric("Lipinski Rules", rules_text.splitlines()[0] if rules_text else "N/A")


# ===================================================================
# TAB 2 — 3D VIEWER
# ===================================================================
with tab_struct:
    st.subheader("🧬 2D / 3D Structure")
    smiles_struct = st.text_input("SMILES for structure view", st.session_state.get("last_smiles", "CCO"), key="smiles_struct")
    col2d, col3d = st.columns(2)
    with col2d:
        st.markdown("### 2D Structure")
        img = draw_molecule(smiles_struct, size=(320, 320))
        st.markdown('<div class="viewer-box">', unsafe_allow_html=True)
        if img:
            st.image(img, width=320)
        st.markdown('</div>', unsafe_allow_html=True)
        if smiles_struct:
            mol = Chem.MolFromSmiles(smiles_struct)
            if mol:
                st.markdown("**Formula:** " + rdMolDescriptors.CalcMolFormula(mol))
                st.markdown(f"**SMILES:** {Chem.MolToSmiles(mol)}")
                st.markdown(f"**HBD/HBA:** {rdMolDescriptors.CalcNumHBD(mol)}/{rdMolDescriptors.CalcNumHBA(mol)}")
    with col3d:
        st.markdown("### 3D Interactive Structure")
        st.markdown('<div class="viewer-box">', unsafe_allow_html=True)
        view = render_3d_molecule(smiles_struct)
        if view:
            render_viewer(view)
        else:
            st.error("3D rendering failed. Try another molecule.")
        st.markdown('</div>', unsafe_allow_html=True)


# ===================================================================
# TAB 5 — OUTPUTS
# ===================================================================
with tab_outputs:
    st.subheader("📊 Model Outputs")
    data = st.session_state.get("pred")
    if data and data.get("valid", False):
        fig = build_bar_chart(data["predictions"])
        if fig:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key="pred_chart_outputs")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### ⚡ Inference Times (s)")
        st.write(data.get("inference_times", {}))
    else:
        st.info("Run a prediction first.")


# ===================================================================
# TAB 6 — DRUG LIKENESS
# ===================================================================
with tab_drug:
    st.subheader("💊 Drug-Likeness")
    data = st.session_state.get("pred")
    if data and data.get("valid", False):
        props = data.get("molecular_properties", {})
        st.text(lipinski_summary(props))
        st.json(props)
    else:
        st.info("Run a prediction first.")


# ===================================================================
# TAB 7 — COMPARE
# ===================================================================
with tab_compare:
    st.subheader("Model Training Comparison")
    root = "experiments/results"
    files = {
        "classical": "classical_gnn_results.json",
        "quantum": "quantum_vqc_full_results.json",
        "hybrid": "hybrid_model_results.json",
    }

    loaded = {}
    for name, fname in files.items():
        path = os.path.join(root, fname)
        if os.path.exists(path):
            try:
                loaded[name] = json.load(open(path))
            except Exception as e:
                st.warning(f"Failed to load {fname}: {e}")

    if loaded:
        cols = st.columns(len(loaded))
        for (name, res), c in zip(loaded.items(), cols):
            with c:
                st.markdown(f"**{name.title()}**")
                st.json(res)
    else:
        st.warning("No training results found in experiments/results.")

    st.markdown("### 🧭 Side-by-Side Comparison")
    sm_left = st.text_input("SMILES A", "CCO", key="cmp_a")
    sm_right = st.text_input("SMILES B", "c1ccccc1", key="cmp_b")
    if st.button("Compare Molecules"):
        try:
            resp_a = get_prediction(sm_left, ["classical", "hybrid"])
            resp_b = get_prediction(sm_right, ["classical", "hybrid"])
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Molecule A**")
                st.image(draw_molecule(sm_left), use_container_width=True)
                st.write(resp_a.get("predictions", {}))
            with colB:
                st.markdown("**Molecule B**")
                st.image(draw_molecule(sm_right), use_container_width=True)
                st.write(resp_b.get("predictions", {}))
        except Exception:
            st.error("Comparison failed. Check SMILES and API.")


# ===================================================================
# TAB 7 — EXAMPLES
# ===================================================================
with tab_examples:
    st.subheader("Try These Inputs")
    examples = [
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
        ("Paracetamol", "CC(=O)NC1=CC=C(O)C=C1O"),
    ]

    for row_start in range(0, len(examples), 3):
        cols = st.columns(3)
        for col, (name, smi) in zip(cols, examples[row_start:row_start+3]):
            with col:
                st.write(f"### {name}")
                img = draw_molecule(smi)
                if img:
                    st.image(img, use_container_width=True)
                if st.button(f"Predict {name}", key=f"btn_{name}"):
                    res = get_prediction(smi, ["classical", "hybrid", "quantum"])
                    st.write(res)

# ===================================================================
# TAB 8 — SIMILARITY
# ===================================================================
with tab_similarity:
    st.subheader("🔍 Molecular Similarity Search")
    query_smiles = st.text_input("Enter query SMILES", "CCO", key="sim_smiles")

    if st.button("Find Similar Molecules"):
        try:
            query_mol = Chem.MolFromSmiles(query_smiles)
            if query_mol is None:
                st.error("Invalid SMILES.")
            else:
                query_fp = Chem.RDKFingerprint(query_mol)
                results = []
                data_path = "data/processed/combined_dataset.csv"
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                    for s in df.get("smiles", [])[:5000]:
                        mol = Chem.MolFromSmiles(s)
                        if mol:
                            fp = Chem.RDKFingerprint(mol)
                            sim = DataStructs.FingerprintSimilarity(query_fp, fp)
                            results.append((s, sim))
                    results = sorted(results, key=lambda x: -x[1])[:10]
                    st.markdown("### 🔥 Top 10 Most Similar Molecules")
                    for smiles, score in results:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            img = Draw.MolToImage(Chem.MolFromSmiles(smiles))
                            st.image(img)
                        with col2:
                            st.write(f"**{smiles}**")
                            st.write(f"Similarity: {score:.3f}")
                else:
                    st.warning("Similarity dataset not found at data/processed/combined_dataset.csv")
        except Exception:
            st.error("Similarity search failed.")


# ===================================================================
# TAB 6 — ABOUT
# ===================================================================
with tab_about:
    st.markdown("""
    ### About This Project
    Hybrid Quantum–Classical ML for molecular property prediction
    using GNNs, Variational Quantum Circuits, and RDKit.
    """)
