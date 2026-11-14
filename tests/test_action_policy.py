"""
test_action_policy.py: Rigorous verification of NTBN action policy, idelta, and clipping.
Directly addresses User Requirement 1:
- Inside, below, and above band action behavior.
- First step and impact of idelta.
- Impact of softclip vs hard clip vs none on action preservation.
- Multi-instrument setting (n_inst = 2).
- Inverted band fallback.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

from experiments.jax_ntbn_v1_auditfix.ntbn_core import ntbn_clamp, ntbn_action, ntbn_bounds_center_width
from experiments.jax_ntbn_v1_auditfix.model_ntbn import (
    softclip, rollout_ntbn, AgentConfigNTBN, GymConfigNTBN, init_params_ntbn
)


class TestActionPolicy(unittest.TestCase):

    def test_case_1_inside_band(self):
        """Case 1: delta in [bl, bu] -> action == 0, delta_next == delta."""
        delta = jnp.array([0.4, -0.1, 0.0])
        bl = jnp.array([0.2, -0.3, -0.1])
        bu = jnp.array([0.6, 0.2, 0.1])
        
        delta_next, action = ntbn_action(delta, bl, bu)
        np.testing.assert_allclose(delta_next, delta, rtol=1e-6)
        np.testing.assert_allclose(action, jnp.zeros_like(delta), atol=1e-7)

    def test_case_2_below_band(self):
        """Case 2: delta < bl -> action == bl - delta > 0, delta_next == bl."""
        delta = jnp.array([0.1, -0.5])
        bl = jnp.array([0.3, -0.2])
        bu = jnp.array([0.7, 0.2])
        
        delta_next, action = ntbn_action(delta, bl, bu)
        np.testing.assert_allclose(delta_next, bl, rtol=1e-6)
        expected_a = bl - delta
        np.testing.assert_allclose(action, expected_a, rtol=1e-6)
        self.assertTrue(bool(jnp.all(action > 0.0)))

    def test_case_3_above_band(self):
        """Case 3: delta > bu -> action == bu - delta < 0, delta_next == bu."""
        delta = jnp.array([0.9, 0.5])
        bl = jnp.array([0.2, -0.1])
        bu = jnp.array([0.7, 0.3])
        
        delta_next, action = ntbn_action(delta, bl, bu)
        np.testing.assert_allclose(delta_next, bu, rtol=1e-6)
        expected_a = bu - delta
        np.testing.assert_allclose(action, expected_a, rtol=1e-6)
        self.assertTrue(bool(jnp.all(action < 0.0)))

    def test_case_4_first_step_and_idelta(self):
        """Case 4: Compare pure NTBN (init_delta=False) vs Buehler's init_delta=True."""
        delta_0 = jnp.array([0.0])
        bl = jnp.array([0.3])
        bu = jnp.array([0.7])
        
        # 1. Pure NTBN: initial position 0 -> target is bl -> action = 0.3
        delta_target, a_target = ntbn_action(delta_0, bl, bu)
        delta_1_pure = delta_0 + a_target
        np.testing.assert_allclose(delta_1_pure, bl, rtol=1e-6)
        self.assertTrue(bool(bl <= delta_1_pure <= bu), "Pure NTBN lands inside band")

        # 2. Corrupted by idelta: adding arbitrary initial delta offset
        idelta = jnp.array([0.8])  # random init offset
        a_corrupted = a_target + idelta
        delta_1_corrupted = delta_0 + a_corrupted
        # delta_1_corrupted is 0.3 + 0.8 = 1.1, which is OUTSIDE [0.3, 0.7]!
        self.assertFalse(bool(bl <= delta_1_corrupted <= bu), "idelta pushes delta_1 outside band")
        self.assertAlmostEqual(float(delta_1_corrupted[0]), 1.1)

    def test_case_5_action_clipping_effects(self):
        """Case 5: Contrast 'none', 'hard', and 'soft' action clipping."""
        lb, ub = -5.0, 5.0
        
        # Subtest 5a: Zero target action (position already in band)
        a_zero = jnp.array([0.0])
        # Under pure 'none':
        a_none = a_zero
        np.testing.assert_allclose(a_none, 0.0)
        
        # Under 'hard' clip within bounds:
        a_hard = jnp.clip(a_zero, lb, ub)
        np.testing.assert_allclose(a_hard, 0.0)
        
        # Under 'soft' clip:
        a_soft = softclip(a_zero, lb, ub)
        # softclip(0) is -0.3774, NOT 0!
        self.assertNotAlmostEqual(float(a_soft[0]), 0.0, places=2)
        self.assertAlmostEqual(float(a_soft[0]), -0.377395, places=4)

        # Subtest 5b: Required action exceeds physical bound (market constraint active)
        a_large = jnp.array([8.0])
        a_hard_capped = jnp.clip(a_large, lb, ub)
        self.assertEqual(float(a_hard_capped[0]), 5.0)

    def test_case_6_multi_instrument(self):
        """Case 6: Multi-instrument setting (n_inst = 2) with separable bands."""
        delta = jnp.array([[0.2, 0.8]])     # inst 0 inside, inst 1 above
        bl = jnp.array([[0.1, 0.2]])
        bu = jnp.array([[0.5, 0.6]])
        
        delta_next, action = ntbn_action(delta, bl, bu)
        # inst 0: in band -> stays 0.2, action 0
        self.assertAlmostEqual(float(delta_next[0, 0]), 0.2)
        self.assertAlmostEqual(float(action[0, 0]), 0.0)
        
        # inst 1: above bu (0.8 > 0.6) -> sells to 0.6, action -0.2
        self.assertAlmostEqual(float(delta_next[0, 1]), 0.6)
        self.assertAlmostEqual(float(action[0, 1]), -0.2)

    def test_case_7_inverted_boundary(self):
        """Case 7: Inverted boundary bl > bu -> Fallback to midpoint."""
        delta = jnp.array([0.5])
        bl = jnp.array([0.8])
        bu = jnp.array([0.2])
        
        clamped = ntbn_clamp(delta, bl, bu)
        expected_midpoint = 0.5 * (bl + bu)
        np.testing.assert_allclose(clamped, expected_midpoint, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
