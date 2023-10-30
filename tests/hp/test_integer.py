import unittest
import numpy as np

from PyHyperparameterSpace.hp.continuous import Integer
from PyHyperparameterSpace.dist.continuous import Uniform


class TestInteger(unittest.TestCase):
    """
    Tests the class Integer.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.bounds = (-10, 11)
        self.default = 1
        self.default2 = np.array([[1, 2], [3, 4]])
        self.default3 = np.array([[0, 0], [0, 0]])
        self.shape2 = (2, 2)
        self.random = np.random.RandomState()
        self.size = 10

        # Tests with all options given
        self.hp = Integer(name=self.name, bounds=self.bounds, default=self.default, shape=self.shape)
        # Tests with all options given (matrix as values)
        self.hp2 = Integer(name=self.name, bounds=self.bounds, default=self.default2, shape=self.shape2)
        # Tests with default=None
        self.hp3 = Integer(name=self.name, bounds=self.bounds, default=None, shape=self.shape2)
        # Test with shape=None
        self.hp4 = Integer(name=self.name, bounds=self.bounds, default=self.default2, shape=None)

    def test_name(self):
        """
        Tests the property name.
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
        self.assertEqual(self.shape2, self.hp2._shape)
        self.assertEqual(self.shape2, self.hp3._shape)
        self.assertEqual(self.shape2, self.hp4._shape)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        self.assertEqual(self.bounds, self.hp._bounds)
        self.assertEqual(self.bounds, self.hp2._bounds)
        self.assertEqual(self.bounds, self.hp3._bounds)
        self.assertEqual(self.bounds, self.hp4._bounds)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default, self.hp._default)
        self.assertTrue(np.all(self.default2 == self.hp2._default))
        self.assertTrue(np.all(self.default3 == self.hp3._default))
        self.assertTrue(np.all(self.default2 == self.hp4._default))

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsInstance(self.hp._distribution, Uniform)
        self.assertIsInstance(self.hp2._distribution, Uniform)
        self.assertIsInstance(self.hp3._distribution, Uniform)
        self.assertIsInstance(self.hp4._distribution, Uniform)

    def test_lb(self):
        """
        Tests the property lb.
        """
        self.assertEqual(self.bounds[0], self.hp.lb)
        self.assertEqual(self.bounds[0], self.hp2.lb)
        self.assertEqual(self.bounds[0], self.hp3.lb)
        self.assertEqual(self.bounds[0], self.hp4.lb)

    def test_ub(self):
        """
        Tests the property ub.
        """
        self.assertEqual(self.bounds[1], self.hp.ub)
        self.assertEqual(self.bounds[1], self.hp2.ub)
        self.assertEqual(self.bounds[1], self.hp3.ub)
        self.assertEqual(self.bounds[1], self.hp4.ub)

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
        self.assertEqual(self.default, self.hp.get_default())
        self.assertTrue(np.all(self.default2 == self.hp2.get_default()))
        self.assertTrue(np.all(self.default3 == self.hp3.get_default()))
        self.assertTrue(np.all(self.default2 == self.hp4.get_default()))

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.hp.get_shape())
        self.assertEqual(self.shape2, self.hp2.get_shape())
        self.assertEqual(self.shape2, self.hp3.get_shape())
        self.assertEqual(self.shape2, self.hp4.get_shape())

    def test_get_bounds(self):
        """
        Tests the method get_bounds().
        """
        self.assertEqual(self.bounds, self.hp.get_bounds())
        self.assertEqual(self.bounds, self.hp2.get_bounds())
        self.assertEqual(self.bounds, self.hp3.get_bounds())
        self.assertEqual(self.bounds, self.hp4.get_bounds())

    def test_get_distribution(self):
        """
        Tests the method get_distribution().
        """
        self.assertIsInstance(self.hp.get_distribution(), Uniform)
        self.assertIsInstance(self.hp2.get_distribution(), Uniform)
        self.assertIsInstance(self.hp3.get_distribution(), Uniform)
        self.assertIsInstance(self.hp4.get_distribution(), Uniform)

    def test_sample(self):
        """
        Tests the method sample().
        """
        sample = self.hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(all(self.bounds[0] <= s < self.bounds[1] for s in sample))

        sample2 = self.hp2.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample2))
        self.assertEqual(self.shape2, sample2[0].shape)
        self.assertTrue(np.all((self.bounds[0] <= sample2) & (sample2 < self.bounds[1])))

        sample3 = self.hp3.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample3))
        self.assertEqual(self.shape2, sample3[0].shape)
        self.assertTrue(np.all((self.bounds[0] <= sample3) & (sample3 < self.bounds[1])))

        sample4 = self.hp4.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample4))
        self.assertEqual(self.shape2, sample4[0].shape)
        self.assertTrue(np.all((self.bounds[0] <= sample4) & (sample4 < self.bounds[1])))

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
