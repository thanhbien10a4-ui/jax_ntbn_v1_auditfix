"""
test_smoke_pipeline.py: Minimal smoke test to confirm end-to-end functionality of NTBN AuditFix.
Adheres strictly to test constraints:
- Only 1 epoch with 64 samples and 2 batches (batch_size=32).
- Pure NTBN configuration (action_clipping='none', init_delta=False).
- Validates that data generation, initialization, JIT compilation, training, validation,
  and history recording execute without runtime errors.
- NOTE: This smoke test is exclusively for pipeline sanity verification and NOT used to claim performance improvement.
"""

import unittest
import sys
import os

# Ensure deephedging-jax is at the very front of sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
jax_repo_path = os.path.join(project_root, "deephedging-jax")

if jax_repo_path in sys.path:
    sys.path.remove(jax_repo_path)
sys.path.insert(0, jax_repo_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from deephedging.world import WorldConfig, simulate
from experiments.jax_ntbn_v1_auditfix.model_ntbn import AgentConfigNTBN, GymConfigNTBN, init_params_ntbn, as_batch
from experiments.jax_ntbn_v1_auditfix.train_ntbn import TrainConfig, train_ntbn
import jax


class TestNTBNSmokePipeline(unittest.TestCase):

    def test_minimal_train_pipeline(self):
        """Runs a 1-epoch smoke test on simulated market data with non-zero transaction cost."""
        # 1. Simulate minimal world (64 samples, 5 steps, cost_s = 0.0002)
        world_cfg = WorldConfig(samples=64, steps=5, cost_s=0.0002, black_scholes=True, seed=42)
        train_world = simulate(world_cfg)
        val_world = simulate(world_cfg)
        
        train_data = as_batch(train_world)
        val_data = as_batch(val_world)
        
        # 2. Configure Pure NTBN Agent & Gym (action_clipping='none', init_delta=False)
        agent_cfg = AgentConfigNTBN(
            features=("price", "time_left"), width=16, depth=2, activation="relu",
            action_clipping="none", init_delta=False
        )
        gym_cfg = GymConfigNTBN(agent=agent_cfg, utility="exp2", lmbda=1.0)
        
        # Feature dimensions
        feat_dims = {k: v.shape[-1] for k, v in train_data["features"].items()}
        key = jax.random.PRNGKey(999)
        params = init_params_ntbn(key, gym_cfg, feat_dims, n_inst=train_world.nInst)
        
        # 3. Configure minimal training: 1 epoch, batch_size=32
        train_cfg = TrainConfig(epochs=1, batch_size=32, learning_rate=0.001, seed=42)
        
        # 4. Execute training
        res = train_ntbn(gym_cfg, train_cfg, params, train_data, val_data)
        
        # 5. Sanity checks
        self.assertIsNotNone(res)
        self.assertEqual(len(res.history["loss"]), 1)
        self.assertEqual(len(res.history["val_loss"]), 1)
        self.assertTrue(res.best_epoch in (0, -1))
        self.assertTrue(all(k in res.history for k in ("loss", "val_loss", "utility", "utility0", "time")))
        print(f"\n[AuditFix Smoke Test OK] 1 Epoch Loss: {res.history['loss'][0]:.6f}, Time: {res.history['time'][0]:.4f}s")


if __name__ == "__main__":
    unittest.main()
