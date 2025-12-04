"""
Hybrid Model Training
Two-stage training strategy for quantum-classical hybrid models
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, List, Tuple, Callable
from pathlib import Path
import json
import time


class HybridTrainer:
    """
    Trainer for hybrid quantum-classical models.

    Training Strategy:
        Stage 1 — Train classical components (freeze quantum)
        Stage 2 — Fine-tune full model (including quantum)
    """

    def __init__(
        self,
        model_forward_fn: Callable,
        optimizer_classical: optax.GradientTransformation,
        optimizer_full: optax.GradientTransformation,
        loss_fn: Callable = None,
        checkpoint_dir: str = "experiments/checkpoints/hybrid",
        seed: int = 42
    ):
        self.model_forward = model_forward_fn
        self.optimizer_classical = optimizer_classical
        self.optimizer_full = optimizer_full
        self.loss_fn = loss_fn or self._mse_loss

        self.rng = jax.random.PRNGKey(seed)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "stage1_train_loss": [],
            "stage1_val_loss": [],
            "stage1_train_mae": [],
            "stage1_val_mae": [],
            "stage2_train_loss": [],
            "stage2_val_loss": [],
            "stage2_train_mae": [],
            "stage2_val_mae": [],
            "epoch_time": []
        }

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        print("✅ Hybrid Trainer initialized")
        print(f"   Checkpoints: {checkpoint_dir}")
        print("   Two-stage training enabled")

    # ============================
    # BASIC METRICS
    # ============================

    @staticmethod
    def _mse_loss(pred, target):
        return jnp.mean((pred - target) ** 2)

    @staticmethod
    def _mae_metric(pred, target):
        return jnp.mean(jnp.abs(pred - target))

    # ============================
    # FREEZE QUANTUM GRADIENTS
    # ============================

    def _zero_quantum_gradients(self, grads):
        """Zero gradients for quantum layer when frozen."""
        if "params" in grads and "quantum_layer" in grads["params"]:
            new_grads = dict(grads)
            new_grads["params"] = dict(grads["params"])
            new_grads["params"]["quantum_layer"] = jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x),
                grads["params"]["quantum_layer"]
            )
            return new_grads
        return grads

    # ============================
    # TRAINING EPOCH
    # ============================

    def train_epoch(
        self,
        params,
        opt_state,
        train_graphs: List[Dict],
        train_targets: jnp.ndarray,
        freeze_quantum: bool = False
    ):
        """One full training epoch."""
        n = len(train_graphs)
        indices = np.random.permutation(n)

        epoch_losses = []
        preds_all = []
        targs_all = []

        # Select optimizer
        optimizer = self.optimizer_classical if freeze_quantum else self.optimizer_full

        for idx in indices:
            graph = train_graphs[idx]
            target = train_targets[idx]

            def loss_fn(p):
                pred = self.model_forward(p, graph, training=True)
                return self.loss_fn(pred, target)

            loss, grads = jax.value_and_grad(loss_fn)(params)

            if freeze_quantum:
                grads = self._zero_quantum_gradients(grads)

            # 🔥 Correct Optax update
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            opt_state = new_opt_state

            epoch_losses.append(float(loss))

            pred_eval = self.model_forward(params, graph, training=False)
            preds_all.append(float(pred_eval))
            targs_all.append(float(target))

        preds_all = jnp.array(preds_all)
        targs_all = jnp.array(targs_all)

        metrics = {
            "loss": float(np.mean(epoch_losses)),
            "mae": float(self._mae_metric(preds_all, targs_all))
        }

        return params, opt_state, metrics

    # ============================
    # VALIDATION
    # ============================

    def evaluate(self, params, graphs, targets):
        preds, losses = [], []

        for g, t in zip(graphs, targets):
            pred = self.model_forward(params, g, training=False)
            loss = self.loss_fn(pred, t)

            preds.append(float(pred))
            losses.append(float(loss))

        preds = jnp.array(preds)
        targets = jnp.array(targets)

        return {
            "loss": float(np.mean(losses)),
            "mae": float(self._mae_metric(preds, targets)),
            "rmse": float(jnp.sqrt(jnp.mean((preds - targets) ** 2))),
            "r2": float(
                1
                - jnp.sum((targets - preds) ** 2)
                / (jnp.sum((targets - jnp.mean(targets)) ** 2) + 1e-8)
            ),
        }

    # ============================
    # CHECKPOINTING
    # ============================

    def save_checkpoint(self, params, epoch, metrics, name="checkpoint"):
        path = self.checkpoint_dir / f"{name}.npz"
        flat, _ = jax.tree_util.tree_flatten(params)
        np.savez(
            path,
            **{f"param_{i}": np.array(p) for i, p in enumerate(flat)},
            epoch=epoch,
            metrics=json.dumps(metrics),
        )

    # ============================
    # FULL TRAINING LOOP
    # ============================

    def fit(
        self,
        params,
        train_graphs,
        train_targets,
        val_graphs,
        val_targets,
        stage1_epochs=30,
        stage2_epochs=20,
        patience=10,
        verbose=True
    ):
        print("\n======================================================")
        print("🏋️  HYBRID MODEL — TWO STAGE TRAINING")
        print("======================================================")

        best_params = params

        # ======================================================
        # 🔵 STAGE 1 — Classical Only
        # ======================================================
        print("\n🔵 STAGE 1 — Train Classical Components")
        print("   (Quantum parameters frozen)")

        opt_state_stage1 = self.optimizer_classical.init(params)
        no_improve = 0

        for epoch in range(stage1_epochs):
            start = time.time()

            params, opt_state_stage1, train_m = self.train_epoch(
                params, opt_state_stage1, train_graphs, train_targets, freeze_quantum=True
            )
            val_m = self.evaluate(params, val_graphs, val_targets)
            dt = time.time() - start

            self.history["stage1_train_loss"].append(train_m["loss"])
            self.history["stage1_val_loss"].append(val_m["loss"])
            self.history["stage1_train_mae"].append(train_m["mae"])
            self.history["stage1_val_mae"].append(val_m["mae"])
            self.history["epoch_time"].append(dt)

            if val_m["loss"] < self.best_val_loss:
                self.best_val_loss = val_m["loss"]
                self.best_epoch = epoch
                best_params = params
                no_improve = 0
                self.save_checkpoint(params, epoch, val_m, "stage1_best")
            else:
                no_improve += 1

            if verbose and (epoch % 5 == 0 or epoch == stage1_epochs - 1):
                print(
                    f"Epoch {epoch+1}/{stage1_epochs} "
                    f"| Train Loss: {train_m['loss']:.4f} "
                    f"| Val Loss: {val_m['loss']:.4f}"
                    f"| Val MAE: {val_m['mae']:.4f}"
                )

            if no_improve >= patience:
                print("⏹️  Early stopping Stage 1")
                break

        print("\n✅ Stage 1 complete!")
        print(f"   Best val MAE: {self.history['stage1_val_mae'][self.best_epoch]:.4f}")

        # Continue with best classical parameters
        params = best_params

        # ======================================================
        # 🟣 STAGE 2 — Full Fine-tuning (Quantum + Classical)
        # ======================================================

        print("\n🟣 STAGE 2 — Fine-tuning Full Hybrid Model")
        opt_state_stage2 = self.optimizer_full.init(params)
        best_stage2_loss = float("inf")
        no_improve = 0

        for epoch in range(stage2_epochs):
            start = time.time()

            params, opt_state_stage2, train_m = self.train_epoch(
                params, opt_state_stage2, train_graphs, train_targets, freeze_quantum=False
            )
            val_m = self.evaluate(params, val_graphs, val_targets)
            dt = time.time() - start

            self.history["stage2_train_loss"].append(train_m["loss"])
            self.history["stage2_val_loss"].append(val_m["loss"])
            self.history["stage2_train_mae"].append(train_m["mae"])
            self.history["stage2_val_mae"].append(val_m["mae"])
            self.history["epoch_time"].append(dt)

            if val_m["loss"] < best_stage2_loss:
                best_stage2_loss = val_m["loss"]
                best_params = params
                no_improve = 0
                self.save_checkpoint(params, epoch, val_m, "stage2_best")
            else:
                no_improve += 1

            if verbose and (epoch % 5 == 0 or epoch == stage2_epochs - 1):
                print(
                    f"Epoch {epoch+1}/{stage2_epochs} "
                    f"| Train Loss: {train_m['loss']:.4f} "
                    f"| Val Loss: {val_m['loss']:.4f}"
                    f"| Val MAE: {val_m['mae']:.4f}"
                )

            if no_improve >= patience:
                print("⏹️  Early stopping Stage 2")
                break

        # Save training history
        hist_path = self.checkpoint_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print("\n🎉 HYBRID TRAINING COMPLETE!")
        print(f"   Best Stage-2 val MAE: {min(self.history['stage2_val_mae'] or [999]):.4f}")

        return best_params, self.history