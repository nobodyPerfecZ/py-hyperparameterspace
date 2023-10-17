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
        self.default_true = True
        self.default_false = False
        self.weights = [0.1, 0.9]
        self.weights_uniform = [0.5, 0.5]
        self.random = np.random.RandomState()
        self.size = 10

        # Tests with all options given
        self.hp = Binary(name=self.name, default=self.default_true, weights=self.weights)
        # Test with default=None
        self.hp2 = Binary(name=self.name, default=None, weights=self.weights)
        # Test with weights=None
        self.hp3 = Binary(name=self.name, default=self.default_true, weights=None)
        # Test with default=None, weights=None
        self.hp4 = Binary(name=self.name, default=None, weights=None)

    def test_name(self):
        """
        Tests the property name
        """
        self.assertEqual(self.name, self.hp._name)
        self.assertEqual(self.name, self.hp2._name)
        self.assertEqual(self.name, self.hp3._name)
        self.assertEqual(self.name, self.hp4._name)

    def test_shape(self):
        """
        Tests the property shape.
        """
        self.assertEqual(self.shape, self.hp._shape)
        self.assertEqual(self.shape, self.hp2._shape)
        self.assertEqual(self.shape, self.hp3._shape)
        self.assertEqual(self.shape, self.hp4._shape)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        self.assertIsNone(self.hp._bounds)
        self.assertIsNone(self.hp2._bounds)
        self.assertIsNone(self.hp3._bounds)
        self.assertIsNone(self.hp4._bounds)

    def test_choices(self):
        """
        Tests the property choices.
        """
        self.assertEqual([True, False], self.hp._choices)
        self.assertEqual([True, False], self.hp2._choices)
        self.assertEqual([True, False], self.hp3._choices)
        self.assertEqual([True, False], self.hp4._choices)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default_true, self.hp._default)
        self.assertEqual(self.default_false, self.hp2._default)
        self.assertEqual(self.default_true, self.hp3._default)
        self.assertEqual(self.default_true, self.hp4._default)

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsInstance(self.hp._distribution, Choice)
        self.assertIsInstance(self.hp2._distribution, Choice)
        self.assertIsInstance(self.hp3._distribution, Choice)
        self.assertIsInstance(self.hp4._distribution, Choice)

    def test_weights(self):
        """
        Tests the property weights.
        """
        self.assertEqual(self.weights, self.hp._weights)
        self.assertEqual(self.weights, self.hp2._weights)
        self.assertEqual(self.weights_uniform, self.hp3._weights)
        self.assertEqual(self.weights_uniform, self.hp4._weights)

    def test_lb(self):
        """
        Tests the property lb.
        """
        self.assertIsNone(self.hp._lb)
        self.assertIsNone(self.hp2._lb)
        self.assertIsNone(self.hp3._lb)
        self.assertIsNone(self.hp4._lb)

    def test_ub(self):
        """
        Tests the property ub.
        """
        self.assertIsNone(self.hp._ub)
        self.assertIsNone(self.hp2._ub)
        self.assertIsNone(self.hp3._ub)
        self.assertIsNone(self.hp4._ub)

    def test_get_name(self):
        """
        Tests the method get_name().
        """
        self.assertEqual(self.name, self.hp.get_name())
        self.assertEqual(self.name, self.hp2.get_name())
        self.assertEqual(self.name, self.hp3.get_name())
        self.assertEqual(self.name, self.hp4.get_name())

    def test_get_default(self):
        """
        Tests the method get_default().
        """
        self.assertEqual(self.default_true, self.hp.get_default())
        self.assertEqual(self.default_false, self.hp2.get_default())
        self.assertEqual(self.default_true, self.hp3.get_default())
        self.assertEqual(self.default_true, self.hp4.get_default())

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.hp.get_shape())
        self.assertEqual(self.shape, self.hp2.get_shape())
        self.assertEqual(self.shape, self.hp3.get_shape())
        self.assertEqual(self.shape, self.hp4.get_shape())

    def test_sample(self):
        """
        Tests the method sample().
        """
        sample = self.hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s is True or s is False for s in sample)

        sample = self.hp2.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s is True or s is False for s in sample)

        sample = self.hp3.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s is True or s is False for s in sample)

        sample = self.hp4.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s is True or s is False for s in sample)

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.hp, self.hp)
        self.assertNotEqual(self.hp, self.hp2)
        self.assertNotEqual(self.hp, self.hp3)
        self.assertNotEqual(self.hp, self.hp4)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertEqual(hash(self.hp), hash(self.hp))
        self.assertNotEqual(hash(self.hp), hash(self.hp2))
        self.assertNotEqual(hash(self.hp), hash(self.hp3))
        self.assertNotEqual(hash(self.hp), hash(self.hp4))


if __name__ == '__main__':
    unittest.main()
