"""Test DynaBINN forward passes for discovery and prediction."""
import unittest
import numpy as np
import torch
from dynabinn.model import UltimatePNet

class TestModelSmoke(unittest.TestCase):
    def test_discovery_and_fixed_topology(self):
        masks = [
            np.ones((6, 5), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
            np.ones((4, 3), dtype=np.float32),
            np.ones((3, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ]
        x = torch.rand(3, 6, 4)
        for discovery in (True, False):
            model = UltimatePNet(6, 4, masks, masks, 3, is_discovery_phase=discovery)
            pred, regularization = model(x)
            self.assertEqual(tuple(pred.shape), (3, 3))
            self.assertTrue(torch.isfinite(pred).all())

if __name__ == "__main__":
    unittest.main()
