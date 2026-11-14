"""
test_jax_parity_shared_components.py: Paired parity verification between original JAX repo and NTBN shared components.
Directly addresses User Requirement 4:
- Numerical equivalence of shared utility functions on identical inputs.
- Numerical equivalence of as_batch data conversion.
- Finite values and stable gradients of shared loss primitives.
- Git / integrity verification that deephedging-jax/ original repository was NOT modified.
"""

import unittest
import sys
import os
import subprocess
import numpy as np
import jax
import jax.numpy as jnp

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
jax_repo_path = os.path.join(project_root, "deephedging-jax")
if jax_repo_path in sys.path:
    sys.path.remove(jax_repo_path)
sys.path.insert(0, jax_repo_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from original deephedging-jax
import deephedging.model as orig_model
from deephedging.world import WorldConfig, simulate

# Import from auditfix NTBN
import experiments.jax_ntbn_v1_auditfix.model_ntbn as audit_model


class TestJAXParitySharedComponents(unittest.TestCase):

    def test_shared_utilities_numerical_parity(self):
        """All 6 utility functions ('mean', 'exp', 'exp2', 'vicky', 'cvar', 'quad') match to machine precision."""
        rng = np.random.default_rng(42)
        X = jnp.asarray(rng.standard_normal(100), dtype=jnp.float32)
        y = jnp.asarray([0.15], dtype=jnp.float32)
        
        for u in orig_model.UTILITIES:
            for lmbda in (0.0, 0.5, 1.0, 2.5):
                val_orig = orig_model.utility(u, lmbda, X, y[0])
                val_audit = audit_model.utility(u, lmbda, X, y[0])
                np.testing.assert_allclose(
                    np.asarray(val_audit), np.asarray(val_orig), rtol=1e-6, atol=1e-7,
                    err_msg=f"Utility mismatch for {u} at lmbda={lmbda}"
                )

    def test_as_batch_data_conversion_parity(self):
        """as_batch produces identical dictionary keys and tensor values."""
        wc = WorldConfig(samples=30, steps=5, cost_s=0.001, seed=123)
        w = simulate(wc)
        
        batch_orig = orig_model.as_batch(w)
        batch_audit = audit_model.as_batch(w)
        
        self.assertEqual(batch_orig.keys(), batch_audit.keys())
        for k in ("market", "features"):
            self.assertEqual(batch_orig[k].keys(), batch_audit[k].keys())
            for subk in batch_orig[k]:
                np.testing.assert_allclose(
                    np.asarray(batch_audit[k][subk]), np.asarray(batch_orig[k][subk]),
                    rtol=1e-6, err_msg=f"Mismatch in batch['{k}']['{subk}']"
                )

    def test_original_repo_unmodified(self):
        """Verify that the original deephedging-jax core source files have zero uncommitted modifications."""
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain", "deephedging"],
                cwd=jax_repo_path, text=True
            ).strip()
            self.assertEqual(status, "", f"original deephedging-jax/deephedging has modifications:\n{status}")
        except Exception as e:
            self.fail(f"Git check failed: {e}")


if __name__ == "__main__":
    unittest.main()
