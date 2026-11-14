"""
test_ntbn_gradient.py: Unit tests for differentiability, gradients, and numerical stability of NTBN (AuditFix Edition).
Verifies:
- Subgradients of clamp with respect to delta, bl, and bu.
- Check non-finite values (NaN, Inf) and verify finite gradients.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

from experiments.jax_ntbn_v1_auditfix.ntbn_core import ntbn_clamp, ntbn_action


class TestNTBNGradient(unittest.TestCase):

    def test_gradient_inside_band(self):
        """When delta is strictly inside (bl < delta < bu), d(clamp)/d(delta) == 1, d(clamp)/d(bl) == 0, d(clamp)/d(bu) == 0."""
        def f(delta, bl, bu):
            return jnp.sum(ntbn_clamp(delta, bl, bu))
            
        delta = jnp.array([0.5])
        bl = jnp.array([0.2])
        bu = jnp.array([0.8])
        
        grad_delta, grad_bl, grad_bu = jax.grad(f, argnums=(0, 1, 2))(delta, bl, bu)
        
        self.assertEqual(float(grad_delta[0]), 1.0)
        self.assertEqual(float(grad_bl[0]), 0.0)
        self.assertEqual(float(grad_bu[0]), 0.0)

    def test_gradient_below_band(self):
        """When delta < bl, d(clamp)/d(delta) == 0, d(clamp)/d(bl) == 1, d(clamp)/d(bu) == 0."""
        def f(delta, bl, bu):
            return jnp.sum(ntbn_clamp(delta, bl, bu))
            
        delta = jnp.array([0.1])
        bl = jnp.array([0.3])
        bu = jnp.array([0.8])
        
        grad_delta, grad_bl, grad_bu = jax.grad(f, argnums=(0, 1, 2))(delta, bl, bu)
        
        self.assertEqual(float(grad_delta[0]), 0.0)
        self.assertEqual(float(grad_bl[0]), 1.0)
        self.assertEqual(float(grad_bu[0]), 0.0)

    def test_gradient_above_band(self):
        """When delta > bu, d(clamp)/d(delta) == 0, d(clamp)/d(bl) == 0, d(clamp)/d(bu) == 1."""
        def f(delta, bl, bu):
            return jnp.sum(ntbn_clamp(delta, bl, bu))
            
        delta = jnp.array([0.9])
        bl = jnp.array([0.2])
        bu = jnp.array([0.7])
        
        grad_delta, grad_bl, grad_bu = jax.grad(f, argnums=(0, 1, 2))(delta, bl, bu)
        
        self.assertEqual(float(grad_delta[0]), 0.0)
        self.assertEqual(float(grad_bl[0]), 0.0)
        self.assertEqual(float(grad_bu[0]), 1.0)

    def test_gradient_inverted_band(self):
        """When bl > bu, fallback to 0.5 * (bl + bu): gradients w.r.t bl and bu should be 0.5."""
        def f(delta, bl, bu):
            return jnp.sum(ntbn_clamp(delta, bl, bu))
            
        delta = jnp.array([0.5])
        bl = jnp.array([0.8])
        bu = jnp.array([0.2])
        
        grad_delta, grad_bl, grad_bu = jax.grad(f, argnums=(0, 1, 2))(delta, bl, bu)
        
        self.assertEqual(float(grad_delta[0]), 0.0)
        self.assertAlmostEqual(float(grad_bl[0]), 0.5, places=5)
        self.assertAlmostEqual(float(grad_bu[0]), 0.5, places=5)

    def test_no_nan_or_inf_in_extremes(self):
        """Verify no NaN or Inf is produced under extreme numerical values."""
        deltas = jnp.array([-1e4, 0.0, 1e4, 1e-6])
        bls = jnp.array([-1e3, -0.5, 500.0, -1e-5])
        bus = jnp.array([1e3, 0.5, 600.0, 1e-5])
        
        out = ntbn_clamp(deltas, bls, bus)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
        
        def loss_dummy(b_raw):
            bl, bu = b_raw[0], b_raw[1]
            return jnp.sum(ntbn_clamp(deltas, bl, bu) ** 2)
            
        b_init = jnp.array([-0.2, 0.2])
        g = jax.grad(loss_dummy)(b_init)
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


if __name__ == "__main__":
    unittest.main()
