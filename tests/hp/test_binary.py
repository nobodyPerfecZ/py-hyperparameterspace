import unittest
import numpy as np

from PyHyperparameterSpace.hp.categorical import Binary
from PyHyperparameterSpace.dist.categorical import Choice


class TestBinary(unittest.TestCase):
    """
    Tests the class Binary.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.default = True
        self.weights = [0.1, 0.9]
        self.random = np.random.RandomState()
        self.size = 10
        self.hp = Binary(name=self.name, default=self.default, weights=self.weights)
        self.hp2 = Binary(name=self.name, default=self.default, weights=None)

    def test_name(self):
        """
        Tests the property name
        """
        self.assertEqual(self.name, self.hp._name)

    def test_shape(self):
        """
        Tests the property shape.
        """
        self.assertEqual(self.shape, self.hp._shape)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        self.assertIsNone(self.hp._bounds)

    def test_choices(self):
        """
        Tests the property choices.
        """
        self.assertEqual([True, False], self.hp._choices)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default, self.hp._default)

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsInstance(self.hp._distribution, Choice)

    def test_weights(self):
        """
        Tests the property weights.
        """
        self.assertEqual(self.weights, self.hp._weights)

    def test_lb(self):
        """
        Tests the property lb.
        """
        self.assertIsNone(self.hp._lb)

    def test_ub(self):
        """
        Tests the property ub.
        """
        self.assertIsNone(self.hp._ub)

    def test_get_name(self):
        """
        Tests the method get_name().
        """
        self.assertEqual(self.name, self.hp.get_name())

    def test_get_default(self):
        """
        Tests the method get_default().
        """
        self.assertEqual(self.default, self.hp.get_default())

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.hp.get_shape())

    def test_sample(self):
        """
        Tests the method sample().
        """
        sample = self.hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s is True or s is False for s in sample)

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.hp, self.hp)
        self.assertNotEqual(self.hp, self.hp2)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertNotEqual(hash(self.hp), hash(self.hp2))


if __name__ == '__main__':
    unittest.main()
