"""
test_ntbn_checkpoint.py: Unit tests for NTBN checkpointing and state persistence (AuditFix Edition).
Verifies:
- Checkpoint path hash generation is deterministic.
- Model state can be serialized, saved to disk, reloaded, and matches exactly.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

from experiments.jax_ntbn_v1_auditfix.model_ntbn import AgentConfigNTBN, GymConfigNTBN, init_params_ntbn
from experiments.jax_ntbn_v1_auditfix.train_ntbn import TrainConfig, checkpoint_path, _save


class TestNTBNCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.gym_cfg = GymConfigNTBN(agent=AgentConfigNTBN(features=("price", "time_left"), width=16, depth=2))
        self.train_cfg = TrainConfig(epochs=10, batch_size=16, learning_rate=0.001)
        self.key = jax.random.PRNGKey(42)
        self.feature_dims = {"price": 1, "time_left": 1}
        self.params = init_params_ntbn(self.key, self.gym_cfg, self.feature_dims, n_inst=1)
        
        # Synthetic data
        self.data = {
            "market": {"payoff": jnp.zeros((32,)), "hedges": jnp.zeros((32, 5, 1))},
            "features": {"price": jnp.ones((32, 5, 1)), "time_left": jnp.zeros((32, 5, 1))}
        }
        self.val_data = self.data

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_path_determinism(self):
        """Checkpoint path is deterministic given identical configs and initial state."""
        p1 = checkpoint_path(self.temp_dir, self.gym_cfg, self.train_cfg, self.params, self.data, self.val_data)
        p2 = checkpoint_path(self.temp_dir, self.gym_cfg, self.train_cfg, self.params, self.data, self.val_data)
        self.assertEqual(p1, p2)
        self.assertTrue(p1.endswith(".pkl"))

    def test_checkpoint_save_and_reload(self):
        """Saving state and reloading reproduces identical tensor values."""
        ckpt_file = os.path.join(self.temp_dir, "test_ckpt.pkl")
        state = {
            "carry": (self.params, {}),
            "best_params": self.params,
            "best_loss": -1.025,
            "best_epoch": 3,
            "history": {"loss": [-1.01, -1.02, -1.025]}
        }
        _save(ckpt_file, state)
        self.assertTrue(os.path.exists(ckpt_file))
        
        import pickle
        with open(ckpt_file, "rb") as f:
            loaded = pickle.load(f)
            
        self.assertEqual(loaded["best_epoch"], 3)
        self.assertAlmostEqual(loaded["best_loss"], -1.025)
        # Check params equality
        orig_leaves = jax.tree_util.tree_leaves(self.params)
        loaded_leaves = jax.tree_util.tree_leaves(loaded["best_params"])
        for o, l in zip(orig_leaves, loaded_leaves):
            np.testing.assert_allclose(np.asarray(o), np.asarray(l), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
