import sys
import types
import unittest
import numpy as np

# Provide a minimal shim for scipy.linalg if SciPy is not installed
if "scipy" not in sys.modules:
    scipy = types.ModuleType("scipy")
    # Map required linalg functions to numpy.linalg
    scipy.linalg = types.SimpleNamespace(
        inv=np.linalg.inv,
        det=np.linalg.det,
    )
    sys.modules["scipy"] = scipy

from utils.gaussian import NdimGaussian
from utils import distances


class TestBhattacharyya(unittest.TestCase):
    def test_bhattacharyya_zero_when_identical(self):
        # Identical Gaussians should yield zero distance (2D case)
        eta = np.array([0.0, 0.0])
        lam = np.eye(2)
        g1 = NdimGaussian(2, eta=eta, lam=lam)
        g2 = NdimGaussian(2, eta=eta.copy(), lam=lam.copy())

        d = distances.bhattacharyya(g1, g2)
        self.assertAlmostEqual(float(d), 0.0, places=12)

    def test_bhattacharyya_symmetric(self):
        # Symmetry: D(g1, g2) == D(g2, g1)
        g1 = NdimGaussian(2, eta=np.array([1.0, -2.0]), lam=np.array([[2.0, 0.1], [0.1, 1.5]]))
        g2 = NdimGaussian(2, eta=np.array([-0.5, 0.5]), lam=np.array([[1.0, 0.2], [0.2, 3.0]]))

        d12 = distances.bhattacharyya(g1, g2)
        d21 = distances.bhattacharyya(g2, g1)
        self.assertAlmostEqual(float(d12), float(d21), places=12)

    def test_bhattacharyya_increases_with_mean_separation(self):
        # For identical covariances, distance should increase with ||delta_mean||
        lam = np.eye(2)
        g_ref = NdimGaussian(2, eta=np.array([0.0, 0.0]), lam=lam)
        g_small = NdimGaussian(2, eta=np.array([0.5, -0.5]), lam=lam)
        g_large = NdimGaussian(2, eta=np.array([2.0, -2.0]), lam=lam)

        d_small = distances.bhattacharyya(g_ref, g_small)
        d_large = distances.bhattacharyya(g_ref, g_large)
        self.assertLess(float(d_small), float(d_large))

    def test_bhattacharyya_positive_for_covariance_mismatch_equal_means(self):
        # With equal means but different covariances, distance > 0
        eta = np.array([0.0, 0.0])
        lam1 = np.eye(2)  # precision = I -> covariance = I
        lam2 = 2.0 * np.eye(2)  # precision = 2I -> covariance = 0.5I

        g1 = NdimGaussian(2, eta=eta, lam=lam1)
        g2 = NdimGaussian(2, eta=eta, lam=lam2)

        d = distances.bhattacharyya(g1, g2)
        self.assertGreater(float(d), 0.0)


class TestMahalanobis(unittest.TestCase):
    def test_mahalanobis_vector_shape_and_values(self):
        # Current implementation returns a vector: norm(delta)/sqrt(diag(lam1))
        eta1 = np.array([0.0, 0.0])
        eta2 = np.array([3.0, 4.0])
        lam1 = np.diag([4.0, 9.0])  # sqrt(diag) = [2, 3]

        g1 = NdimGaussian(2, eta=eta1, lam=lam1)
        g2 = NdimGaussian(2, eta=eta2, lam=np.eye(2))

        d = distances.mahalanobis(g1, g2)
        expected = np.array([np.linalg.norm(eta1 - eta2) / 2.0, np.linalg.norm(eta1 - eta2) / 3.0])
        self.assertEqual(d.shape, expected.shape)
        self.assertTrue(np.allclose(d, expected))

    def test_mahalanobis_zero_when_same_mean(self):
        eta = np.array([1.2, -3.4])
        lam = np.diag([1.0, 5.0])
        g1 = NdimGaussian(2, eta=eta, lam=lam)
        g2 = NdimGaussian(2, eta=eta.copy(), lam=lam.copy())

        d = distances.mahalanobis(g1, g2)
        self.assertTrue(np.allclose(d, np.zeros(2)))


class TestBhattacharyya1D(unittest.TestCase):
    def test_bhattacharyya_1d_zero_when_identical(self):
        # Identical 1D Gaussians should yield zero distance
        eta = np.array([0.0])
        lam = np.array([[1.0]])
        g1 = NdimGaussian(1, eta=eta, lam=lam)
        g2 = NdimGaussian(1, eta=eta.copy(), lam=lam.copy())

        d = distances.bhattacharyya(g1, g2)
        self.assertAlmostEqual(float(d), 0.0, places=12)

    def test_bhattacharyya_1d_increases_with_mean_separation(self):
        # For identical precisions, distance should increase with |delta_mean| in 1D
        lam = np.array([[2.0]])
        g_ref = NdimGaussian(1, eta=np.array([0.0]), lam=lam)
        g_small = NdimGaussian(1, eta=np.array([0.5]), lam=lam)
        g_large = NdimGaussian(1, eta=np.array([2.0]), lam=lam)

        d_small = distances.bhattacharyya(g_ref, g_small)
        d_large = distances.bhattacharyya(g_ref, g_large)
        self.assertLess(float(d_small), float(d_large))


class TestBhattacharyyaEqualCov(unittest.TestCase):
    def test_bhattacharyya_equal_covariances_reduces_to_quadratic(self):
        # If precisions are equal (covariances equal), the log-determinant term cancels
        lam = np.array([[3.0, 0.2], [0.2, 2.0]])
        g1 = NdimGaussian(2, eta=np.array([1.0, -1.0]), lam=lam)
        g2 = NdimGaussian(2, eta=np.array([-2.0, 0.5]), lam=lam.copy())

        cov = np.linalg.inv(lam)
        delta = g1.eta - g2.eta
        expected = 0.125 * (delta.T @ cov @ delta)
        d = distances.bhattacharyya(g1, g2)
        self.assertAlmostEqual(float(d), float(expected), places=12)


class TestMahalanobisMore(unittest.TestCase):
    def test_mahalanobis_ignores_off_diagonal(self):
        # Only diag(lam1) is used, so off-diagonal elements should not affect the result
        eta1 = np.array([0.0, 0.0])
        eta2 = np.array([3.0, 4.0])
        lam_diag = np.diag([4.0, 9.0])
        lam_full = np.array([[4.0, 0.5], [0.5, 9.0]])  # same diag as lam_diag

        g1_diag = NdimGaussian(2, eta=eta1, lam=lam_diag)
        g1_full = NdimGaussian(2, eta=eta1, lam=lam_full)
        g2 = NdimGaussian(2, eta=eta2, lam=np.eye(2))

        d_diag = distances.mahalanobis(g1_diag, g2)
        d_full = distances.mahalanobis(g1_full, g2)
        self.assertTrue(np.allclose(d_diag, d_full))

    def test_mahalanobis_inf_when_zero_on_diag(self):
        # Division by zero on sqrt(diag(lam1)) should yield inf for those components
        eta1 = np.array([0.0, 0.0, 0.0])
        eta2 = np.array([3.0, 4.0, 12.0])
        lam1 = np.diag([1.0, 0.0, 4.0])  # second diagonal is zero

        g1 = NdimGaussian(3, eta=eta1, lam=lam1)
        g2 = NdimGaussian(3, eta=eta2, lam=np.eye(3))

        d = distances.mahalanobis(g1, g2)
        # Check inf at index 1 and finite elsewhere
        self.assertTrue(np.isfinite(d[0]))
        self.assertTrue(np.isinf(d[1]))
        self.assertTrue(np.isfinite(d[2]))


class TestBhattacharyyaPrecisionScaling(unittest.TestCase):
    def test_bhattacharyya_decreases_with_higher_precision_when_equal(self):
        # With equal precisions, the distance is 0.125 * delta^T * cov * delta, cov = inv(lam)
        # Increasing precision (lam) decreases covariance, hence decreases distance for fixed delta
        delta = np.array([1.0, -2.0])
        eta1 = np.array([0.0, 0.0])
        eta2 = eta1 + delta

        lam_low = np.eye(2)          # cov = I
        lam_high = 4.0 * np.eye(2)   # cov = 0.25 I

        g1_low = NdimGaussian(2, eta=eta1, lam=lam_low)
        g2_low = NdimGaussian(2, eta=eta2, lam=lam_low.copy())

        g1_high = NdimGaussian(2, eta=eta1, lam=lam_high)
        g2_high = NdimGaussian(2, eta=eta2, lam=lam_high.copy())

        d_low = distances.bhattacharyya(g1_low, g2_low)
        d_high = distances.bhattacharyya(g1_high, g2_high)
        self.assertLess(float(d_high), float(d_low))


class TestBhattacharyya1DCovMismatch(unittest.TestCase):
    def test_bhattacharyya_1d_positive_for_covariance_mismatch_equal_means(self):
        # 1D equal means but different precisions => positive distance
        eta = np.array([0.0])
        lam1 = np.array([[1.0]])
        lam2 = np.array([[4.0]])

        g1 = NdimGaussian(1, eta=eta, lam=lam1)
        g2 = NdimGaussian(1, eta=eta.copy(), lam=lam2)

        d = distances.bhattacharyya(g1, g2)
        self.assertGreater(float(d), 0.0)


class TestInputImmutability(unittest.TestCase):
    def test_bhattacharyya_does_not_modify_inputs(self):
        # Ensure that calling distance functions does not mutate the Gaussian parameters
        eta = np.array([1.0, -2.0])
        lam = np.array([[2.0, 0.0], [0.0, 3.0]])

        g1 = NdimGaussian(2, eta=eta.copy(), lam=lam.copy())
        g2 = NdimGaussian(2, eta=(eta + 0.5), lam=(lam + np.eye(2)))

        eta1_before = g1.eta.copy()
        lam1_before = g1.lam.copy()

        _ = distances.bhattacharyya(g1, g2)
        _ = distances.mahalanobis(g1, g2)

        self.assertTrue(np.allclose(g1.eta, eta1_before))
        self.assertTrue(np.allclose(g1.lam, lam1_before))


class TestMahalanobisScaling(unittest.TestCase):
    def test_mahalanobis_inversely_scales_with_sqrt_precision(self):
        # For fixed delta, d = ||delta|| / sqrt(diag(lam1)) component-wise
        eta1 = np.array([0.0, 0.0])
        eta2 = np.array([3.0, 4.0])
        delta_norm = np.linalg.norm(eta2 - eta1)

        lam_a = np.diag([1.0, 1.0])
        lam_b = np.diag([4.0, 9.0])

        g1_a = NdimGaussian(2, eta=eta1, lam=lam_a)
        g1_b = NdimGaussian(2, eta=eta1, lam=lam_b)
        g2 = NdimGaussian(2, eta=eta2, lam=np.eye(2))

        d_a = distances.mahalanobis(g1_a, g2)
        d_b = distances.mahalanobis(g1_b, g2)

        # Expected component-wise scaling factors: sqrt([1,1]) / sqrt([4,9]) = [1/2, 1/3]
        expected = np.array([delta_norm * 0.5, delta_norm * (1.0/3.0)])
        self.assertTrue(np.allclose(d_b, expected))


class TestDimensionMismatch(unittest.TestCase):
    def test_bhattacharyya_raises_on_dimension_mismatch(self):
        # Mismatched dimensions should raise due to incompatible shapes
        g1 = NdimGaussian(2, eta=np.array([0.0, 0.0]), lam=np.eye(2))
        g2 = NdimGaussian(3, eta=np.array([0.0, 0.0, 0.0]), lam=np.eye(3))
        with self.assertRaises(Exception):
            _ = distances.bhattacharyya(g1, g2)

    def test_mahalanobis_raises_on_dimension_mismatch(self):
        g1 = NdimGaussian(2, eta=np.array([0.0, 0.0]), lam=np.eye(2))
        g2 = NdimGaussian(3, eta=np.array([0.0, 0.0, 0.0]), lam=np.eye(3))
        with self.assertRaises(Exception):
            _ = distances.mahalanobis(g1, g2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
