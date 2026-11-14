"""
test_cost_units_and_invariants.py: Rigorous verification of cost formulas, units, and accounting invariants.
Directly addresses User Requirement 2:
- Verification of market["cost"] == S_t * cost_s under time-varying spot prices.
- Verification of PnL == sum(a_t * (S_T - S_t)).
- Verification that gains == payoff + pnl - cost in dollar units.
- Verification of sign conventions and timing of cash flows.
"""

import unittest
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp

# Ensure deephedging-jax is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
jax_repo_path = os.path.join(project_root, "deephedging-jax")
if jax_repo_path in sys.path:
    sys.path.remove(jax_repo_path)
sys.path.insert(0, jax_repo_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from deephedging.world import WorldConfig, simulate
from experiments.jax_ntbn_v1_auditfix.model_ntbn import (
    AgentConfigNTBN, GymConfigNTBN, init_params_ntbn, rollout_ntbn, as_batch
)


class TestCostUnitsAndInvariants(unittest.TestCase):

    def setUp(self):
        # Time-varying spot price world: 20 steps, drift = 0.1, rvol = 0.25, cost_s = 0.002 (20 bps)
        self.cost_s = 0.002
        self.wc = WorldConfig(
            samples=20, steps=10, cost_s=self.cost_s, black_scholes=True,
            rvol=0.25, drift=0.1, seed=777
        )
        self.world = simulate(self.wc)
        self.batch = as_batch(self.world)

    def test_cost_array_contains_spot_price(self):
        """Verify that market['cost'] is time-varying and identically equals spot * cost_s."""
        spot_prices = self.world.features["spot"][:, :, 0]
        market_costs = self.world.market["cost"][:, :, 0]
        
        # Verify that spot prices actually vary over time (not constant 1.0)
        spot_std = np.std(spot_prices, axis=1)
        self.assertTrue(np.all(spot_std > 0.01), "Spot prices must vary over time")
        
        # Check exact equivalence: cost == spot * cost_s
        expected_cost = spot_prices * self.cost_s
        np.testing.assert_allclose(market_costs, expected_cost, rtol=1e-6)

    def test_hedges_array_is_terminal_difference(self):
        """Verify that market['hedges'] is S_T - S_t."""
        spot_prices = self.world.features["spot"][:, :, 0]
        terminal_spots = self.world.spot_all[:, -1]
        market_hedges = self.world.market["hedges"][:, :, 0]
        
        expected_hedges = terminal_spots[:, None] - spot_prices
        np.testing.assert_allclose(market_hedges, expected_hedges, rtol=1e-6)

    def test_rollout_cost_and_pnl_accounting_invariants(self):
        """Verify accounting identity: gains == payoff + pnl - cost in identical dollar units."""
        agent_cfg = AgentConfigNTBN(features=("price", "time_left"), width=16, depth=2, action_clipping="none")
        gym_cfg = GymConfigNTBN(agent=agent_cfg, utility="exp2", lmbda=1.0)
        
        feat_dims = {k: v.shape[-1] for k, v in self.batch["features"].items()}
        key = jax.random.PRNGKey(123)
        params = init_params_ntbn(key, gym_cfg, feat_dims, n_inst=self.world.nInst)
        
        res = rollout_ntbn(params, gym_cfg, self.batch)
        
        actions = np.asarray(res["actions"])[:, :, 0]  # (n_samples, n_steps)
        hedges = np.asarray(self.batch["market"]["hedges"])[:, :, 0]
        costs = np.asarray(self.batch["market"]["cost"])[:, :, 0]
        payoff = np.asarray(res["payoff"])
        
        # Analytical PnL: sum(a_t * (S_T - S_t))
        expected_pnl = np.sum(actions * hedges, axis=1)
        # Analytical Cost: sum(|a_t| * S_t * cost_s)
        expected_cost = np.sum(np.abs(actions) * costs, axis=1)
        
        np.testing.assert_allclose(np.asarray(res["pnl"]), expected_pnl, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(res["cost"]), expected_cost, rtol=1e-5)
        
        # Verify gains identity: gains == payoff + pnl - cost
        expected_gains = payoff + expected_pnl - expected_cost
        np.testing.assert_allclose(np.asarray(res["gains"]), expected_gains, rtol=1e-5)
        
        # Verify sign conventions:
        # 1. Cost is always non-negative: cost >= 0
        self.assertTrue(np.all(np.asarray(res["cost"]) >= -1e-7))
        # 2. Derivative liability payoff <= 0
        self.assertTrue(np.all(payoff <= 1e-7))


if __name__ == "__main__":
    unittest.main()
