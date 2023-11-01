import unittest
import numpy as np

from PyHyperparameterSpace.hp.abstract_hp import Hyperparameter


class TestHyperparameter(unittest.TestCase):
    """
    Tests the methods of the abstract class Hyperparameter.
    """

    def test_get_sample_size(self):
        """
        Tests the class method get_sample_size().
        """
        sample_size = Hyperparameter._get_sample_size(None, (2,))
        sample_size2 = Hyperparameter._get_sample_size(1, (1,))
        sample_size3 = Hyperparameter._get_sample_size(2, (2,))
        sample_size4 = Hyperparameter._get_sample_size(10, (2, 2))
        sample_size5 = Hyperparameter._get_sample_size(10, (2, 2))
        sample_size6 = Hyperparameter._get_sample_size(2, 2)

        self.assertEqual((2,), sample_size)
        self.assertEqual(1, sample_size2)
        self.assertEqual((2, 2), sample_size3)
        self.assertEqual((10, 2, 2), sample_size4)
        self.assertEqual((10, 2, 2), sample_size5)
        self.assertEqual((2, 2), sample_size6)

    def test_normalize(self):
        """
        Tests the class method normalize().
        """
        prob_dist = Hyperparameter._normalize([1, 2, 3, 4, 5])
        prob_dist2 = Hyperparameter._normalize([0.1, 0.2, 0.3, 0.4])
        prob_dist3 = Hyperparameter._normalize(np.array([1, 2, 3, 4, 5]))
        prob_dist4 = Hyperparameter._normalize(np.array([0.1, 0.2, 0.3, 0.4]))

        self.assertIsInstance(prob_dist, list)
        self.assertIsInstance(prob_dist, list)
        self.assertIsInstance(prob_dist3, np.ndarray)
        self.assertIsInstance(prob_dist4, np.ndarray)
        self.assertEqual(1, sum(prob_dist))
        self.assertEqual(1, sum(prob_dist2))
        self.assertEqual(1, sum(prob_dist3))
        self.assertEqual(1, sum(prob_dist4))


if __name__ == '__main__':
    unittest.main()
