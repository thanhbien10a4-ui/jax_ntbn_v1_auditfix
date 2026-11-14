"""
test_ntbn_clamp.py: Unit tests for NTBN clamp function and boundary conditions (AuditFix Edition).
Verifies:
- Delta inside band: delta_next == delta (no transaction)
- Delta below band: delta_next == bl (buy to lower boundary)
- Delta above band: delta_next == bu (sell to upper boundary)
- Inverted boundary (bl > bu): fallback to (bl + bu) / 2
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

from experiments.jax_ntbn_v1_auditfix.ntbn_core import ntbn_clamp, ntbn_action, ntbn_bounds_center_width


class TestNTBNClamp(unittest.TestCase):

    def test_inside_band_no_transaction(self):
        """Case 1: Position is inside the band [bl, bu] -> Hold position, zero action."""
        delta = jnp.array([0.5, 0.2, -0.1, 0.0])
        bl = jnp.array([0.2, 0.0, -0.3, -0.2])
        bu = jnp.array([0.8, 0.4, 0.1, 0.2])
        
        clamped = ntbn_clamp(delta, bl, bu)
        delta_next, action = ntbn_action(delta, bl, bu)
        
        np.testing.assert_allclose(clamped, delta, rtol=1e-6)
        np.testing.assert_allclose(delta_next, delta, rtol=1e-6)
        np.testing.assert_allclose(action, jnp.zeros_like(delta), atol=1e-7)

    def test_below_band_buy_to_lower(self):
        """Case 2: Position is below lower bound bl -> Buy up to bl."""
        delta = jnp.array([0.1, -0.5, 0.0])
        bl = jnp.array([0.3, -0.2, 0.4])
        bu = jnp.array([0.7, 0.2, 0.8])
        
        clamped = ntbn_clamp(delta, bl, bu)
        delta_next, action = ntbn_action(delta, bl, bu)
        
        np.testing.assert_allclose(clamped, bl, rtol=1e-6)
        np.testing.assert_allclose(delta_next, bl, rtol=1e-6)
        expected_action = bl - delta
        np.testing.assert_allclose(action, expected_action, rtol=1e-6)
        self.assertTrue(bool(jnp.all(action > 0.0)))

    def test_above_band_sell_to_upper(self):
        """Case 3: Position is above upper bound bu -> Sell down to bu."""
        delta = jnp.array([0.9, 0.5, 1.2])
        bl = jnp.array([0.2, -0.1, 0.3])
        bu = jnp.array([0.7, 0.3, 1.0])
        
        clamped = ntbn_clamp(delta, bl, bu)
        delta_next, action = ntbn_action(delta, bl, bu)
        
        np.testing.assert_allclose(clamped, bu, rtol=1e-6)
        np.testing.assert_allclose(delta_next, bu, rtol=1e-6)
        expected_action = bu - delta
        np.testing.assert_allclose(action, expected_action, rtol=1e-6)
        self.assertTrue(bool(jnp.all(action < 0.0)))

    def test_boundary_exact_hits(self):
        """Case 4: Position is exactly on bl or bu -> Hold position, zero action."""
        bl = jnp.array([0.2, -0.4])
        bu = jnp.array([0.8, 0.4])
        delta = jnp.array([0.2, 0.4])
        
        clamped = ntbn_clamp(delta, bl, bu)
        delta_next, action = ntbn_action(delta, bl, bu)
        
        np.testing.assert_allclose(clamped, delta, rtol=1e-6)
        np.testing.assert_allclose(action, jnp.zeros_like(delta), atol=1e-7)

    def test_inverted_boundary_midpoint(self):
        """Case 5: Inverted boundary bl > bu -> Fallback to (bl + bu) / 2."""
        delta = jnp.array([0.5, 0.0, 1.0])
        bl = jnp.array([0.8, 0.5, 0.7])  # Inverted: bl > bu
        bu = jnp.array([0.4, 0.1, 0.3])
        
        clamped = ntbn_clamp(delta, bl, bu)
        expected_midpoint = 0.5 * (bl + bu)
        
        np.testing.assert_allclose(clamped, expected_midpoint, rtol=1e-6)

    def test_center_width_parameterization(self):
        """Case 6: Center and half-width parameterization guarantees bl <= bu strictly."""
        raw_out = jnp.array([
            [0.5, -2.0],   # Center 0.5, raw width -2.0
            [-0.3, 5.0],   # Center -0.3, raw width 5.0
            [1.2, 0.0],    # Center 1.2, raw width 0.0
        ])
        bl, bu = ntbn_bounds_center_width(raw_out, n_inst=1)
        
        # Verify bu >= bl for all entries
        self.assertTrue(bool(jnp.all(bu >= bl)))
        # Verify half-width is non-negative
        np.testing.assert_allclose(0.5 * (bu + bl), raw_out[:, :1], rtol=1e-6)
        half_w = 0.5 * (bu - bl)
        self.assertTrue(bool(jnp.all(half_w >= 0.0)))


if __name__ == "__main__":
    unittest.main()
