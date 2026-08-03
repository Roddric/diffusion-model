"""Validation-selected finite forecast pools."""

import numpy as np

from research.phase2f_pooling import _linear_pool


def test_linear_pool_respects_boundary_weights():
    diffusion = np.full((4, 2, 3), 2.0)
    baseline = np.full((4, 2, 3), -1.0)

    np.testing.assert_allclose(
        _linear_pool(diffusion, baseline, 0.0), baseline
    )
    np.testing.assert_allclose(
        _linear_pool(diffusion, baseline, 1.0), diffusion
    )
    pooled = _linear_pool(diffusion, baseline, 0.5)
    np.testing.assert_allclose(pooled[:2], 2.0)
    np.testing.assert_allclose(pooled[2:], -1.0)

