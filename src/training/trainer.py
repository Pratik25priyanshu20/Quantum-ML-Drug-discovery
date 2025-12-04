#src/training/trainer.py
"""
Training Infrastructure for JAX Models
Optimized for molecular property prediction
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Optional, Callable
import numpy as np
import time
from pathlib import Path
import json


# ======================================================
# ✔ Metric Helpers
# ======================================================

def mse_loss(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((preds - targets) ** 2)


def mae_metric(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(preds - targets))


def r2_metric(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    ss_res = jnp.sum((targets - preds) ** 2)
    ss_tot = jnp.sum((targets - jnp.mean(targets)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


# ======================================================
# ✔ Training Metrics Tracker
# ======================================================

class TrainingMetrics:
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_r2": [],
            "val_r2": [],
            "epoch_time": []
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))

    def save(self, out: str):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "history": self.history,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch
            }, f, indent=2)


# ======================================================
# ✔ Main Trainer Class
# ======================================================

class GNNTrainer:
    """
    Trainer for GNNs (JAX + Flax)
    Fully compatible with dropout RNG, early stopping,
    and new parameter checkpoint format.
    """

    def __init__(
        self,
        model_forward_fn: Callable,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable = mse_loss,
        metrics_fns: Optional[Dict[str, Callable]] = None,
        checkpoint_dir: str = "experiments/checkpoints",
        seed: int = 42
    ):
        self.model_forward = model_forward_fn
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.metrics_fns = metrics_fns or {
            "mae": mae_metric,
            "r2": r2_metric
        }

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rng = jax.random.PRNGKey(seed)
        self.metrics = TrainingMetrics()

        self.train_step_jit = jax.jit(self._train_step)
        self.eval_step_jit = jax.jit(self._eval_step)

    # ==================================================
    # ✔ Training Step (JIT)
    # ==================================================
    def _train_step(self, params, opt_state, graph, target, rng):
        def loss_fn(p):
            pred = self.model_forward(
                p, graph, training=True, rngs={"dropout": rng}
            )
            return self.loss_fn(pred, target)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ==================================================
    # ✔ Eval Step
    # ==================================================
    def _eval_step(self, params, graph, target):
        pred = self.model_forward(params, graph, training=False, rngs={})
        loss = self.loss_fn(pred, target)
        return pred, loss

    # ==================================================
    # ✔ Train 1 Epoch
    # ==================================================
    def train_epoch(self, params, opt_state, train_graphs, train_targets):
        n = len(train_graphs)
        indices = np.random.permutation(n)

        losses, preds, targs = [], [], []

        for idx in indices:
            graph = train_graphs[idx]
            target = train_targets[idx]

            self.rng, rng_step = jax.random.split(self.rng)

            params, opt_state, loss = self.train_step_jit(
                params, opt_state, graph, target, rng_step
            )
            losses.append(float(loss))

            p = self.model_forward(params, graph, training=False, rngs={})
            preds.append(float(p))
            targs.append(float(target))

        preds = jnp.array(preds)
        targs = jnp.array(targs)

        return params, opt_state, {
            "loss": np.mean(losses),
            "mae": float(mae_metric(preds, targs)),
            "r2": float(r2_metric(preds, targs)),
        }

    # ==================================================
    # ✔ Validation
    # ==================================================
    def evaluate(self, params, eval_graphs, eval_targets):
        losses, preds, targs = [], [], []

        for i in range(len(eval_graphs)):
            p, loss = self.eval_step_jit(
                params, eval_graphs[i], eval_targets[i]
            )
            preds.append(float(p))
            losses.append(float(loss))
            targs.append(float(eval_targets[i]))

        preds = jnp.array(preds)
        targs = jnp.array(targs)

        return {
            "loss": np.mean(losses),
            "mae": float(mae_metric(preds, targs)),
            "r2": float(r2_metric(preds, targs)),
        }

    # ==================================================
    # ✔ Save Checkpoint (FINAL FIX)
    # ==================================================
    def save_checkpoint(self, params, is_best=False):
        leaves, _ = jax.tree_util.tree_flatten(params)

        ckpt = {}
        for i, tensor in enumerate(leaves):
            ckpt[f"param_{i}"] = np.array(tensor)

        ckpt["metrics"] = json.dumps(self.metrics.history)

        name = "best.npz" if is_best else "final.npz"
        path = self.checkpoint_dir / name
        np.savez(path, **ckpt)

        print(f"💾 Saved checkpoint: {path}")

    # ==================================================
    # ✔ Fit Loop
    # ==================================================
    def fit(
        self,
        params,
        train_graphs, train_targets,
        val_graphs, val_targets,
        num_epochs=100, patience=20,
        verbose=True
    ):
        opt_state = self.optimizer.init(params)
        best_params = params
        wait = 0

        if verbose:
            print(f"\n🏋️ Training GNN for {num_epochs} epochs...\n")

        for epoch in range(num_epochs):
            t0 = time.time()

            params, opt_state, train_m = self.train_epoch(
                params, opt_state, train_graphs, train_targets
            )
            val_m = self.evaluate(params, val_graphs, val_targets)

            epoch_time = time.time() - t0

            self.metrics.update({
                "train_loss": train_m["loss"],
                "val_loss": val_m["loss"],
                "train_mae": train_m["mae"],
                "val_mae": val_m["mae"],
                "train_r2": train_m["r2"],
                "val_r2": val_m["r2"],
                "epoch_time": epoch_time
            })

            if val_m["loss"] < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_m["loss"]
                self.metrics.best_epoch = epoch
                best_params = params
                self.save_checkpoint(best_params, is_best=True)
                wait = 0
            else:
                wait += 1

            if verbose and (epoch % 5 == 0 or epoch == num_epochs - 1):
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_m['loss']:.4f} | "
                    f"Val Loss: {val_m['loss']:.4f} | "
                    f"Val MAE: {val_m['mae']:.4f} | "
                    f"Val R2: {val_m['r2']:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            if wait >= patience:
                print(
                    f"\n⛔ Early Stopping at epoch {epoch+1}"
                    f" (Best Epoch: {self.metrics.best_epoch+1})"
                )
                break

        self.save_checkpoint(params, is_best=False)
        self.metrics.save(self.checkpoint_dir / "metrics.json")

        return best_params, opt_state, self.metrics