import unittest
import numpy as np

from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.dist.continuous import Normal
from PyHyperparameterSpace.dist.continuous import Uniform


class TestFloat(unittest.TestCase):
    """
    Tests the class Float.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.bounds = (-12.5, 12.5)
        self.default = 0.5
        self.random = np.random.RandomState()
        self.size = 10
        self.normal_distribution = Normal(loc=0.0, scale=10.0)
        self.uniform_distribution = Uniform()
        self.normal_hp = Float(name=self.name, shape=self.shape, bounds=self.bounds, default=self.default,
                               distribution=self.normal_distribution)
        self.uniform_hp = Float(name=self.name, shape=self.shape, bounds=self.bounds, default=self.default,
                                distribution=self.uniform_distribution)

    def test_name(self):
        """
        Tests the property name.
        """
        self.assertEqual(self.name, self.normal_hp._name)
        self.assertEqual(self.name, self.uniform_hp._name)

    def test_shape(self):
        """
        Tests the property shape.
        """
        self.assertEqual(self.shape, self.normal_hp._shape)
        self.assertEqual(self.shape, self.uniform_hp._shape)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        self.assertEqual(self.bounds, self.normal_hp._bounds)
        self.assertEqual(self.bounds, self.uniform_hp._bounds)

    def test_choices(self):
        """
        Tests the property choices.
        """
        self.assertIsNone(self.normal_hp._choices)
        self.assertIsNone(self.uniform_hp._choices)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default, self.normal_hp._default)
        self.assertEqual(self.default, self.uniform_hp._default)

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsInstance(self.normal_hp._distribution, Normal)
        self.assertIsInstance(self.uniform_hp._distribution, Uniform)

    def test_weights(self):
        """
        Tests the property weights.
        """
        self.assertIsNone(self.normal_hp._weights)
        self.assertIsNone(self.uniform_hp._weights)

    def test_lb(self):
        """
        Tests the property lb.
        """
        self.assertEqual(self.bounds[0], self.normal_hp._lb)
        self.assertEqual(self.bounds[0], self.uniform_hp._lb)

    def test_ub(self):
        """
        Tests the property ub.
        """
        self.assertEqual(self.bounds[1], self.normal_hp._ub)
        self.assertEqual(self.bounds[1], self.uniform_hp._ub)

    def test_get_name(self):
        """
        Tests the method get_name().
        """
        self.assertEqual(self.name, self.normal_hp.get_name())
        self.assertEqual(self.name, self.uniform_hp.get_name())

    def test_get_default(self):
        """
        Tests the method get_default().
        """
        self.assertEqual(self.default, self.normal_hp.get_default())
        self.assertEqual(self.default, self.uniform_hp.get_default())

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.normal_hp.get_shape())
        self.assertEqual(self.shape, self.uniform_hp.get_shape())

    def test_sample(self):
        """
        Tests the method sample().
        """

        normal_sample = self.normal_hp.sample(random=self.random, size=self.size)
        uniform_sample = self.uniform_hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(normal_sample))
        self.assertEqual(self.size, len(uniform_sample))
        self.assertTrue(all(self.bounds[0] <= s < self.bounds[1] for s in normal_sample))
        self.assertTrue(all(self.bounds[0] <= s < self.bounds[1] for s in uniform_sample))

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.normal_hp, self.normal_hp)
        self.assertNotEqual(self.normal_hp, self.uniform_hp)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertNotEqual(hash(self.normal_hp), hash(self.uniform_hp))


class TestFloatMatrix(unittest.TestCase):
    """
    Tests the class Float with matrix as values.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (2, 2)
        self.bounds = (-12.5, 12.5)
        self.default = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.random = np.random.RandomState()
        self.size = 10
        self.normal_distribution = Normal(loc=0.0, scale=10.0)
        self.uniform_distribution = Uniform()
        self.normal_hp = Float(name=self.name, shape=self.shape, bounds=self.bounds, default=self.default,
                               distribution=self.normal_distribution)
        self.uniform_hp = Float(name=self.name, shape=self.shape, bounds=self.bounds, default=self.default,
                                distribution=self.uniform_distribution)

    def test_sample(self):
        """
        Tests the method sample().
        """
        normal_sample = self.normal_hp.sample(random=self.random, size=self.size)
        uniform_sample = self.uniform_hp.sample(random=self.random, size=self.size)

        self.assertEqual(self.size, len(normal_sample))
        self.assertEqual(self.size, len(uniform_sample))
        self.assertEqual(self.shape, normal_sample[0].shape)
        self.assertEqual(self.shape, uniform_sample[0].shape)
        self.assertTrue(np.all((self.bounds[0] <= normal_sample) & (normal_sample < self.bounds[1])))
        self.assertTrue(np.all((self.bounds[0] <= uniform_sample) & (uniform_sample < self.bounds[1])))


if __name__ == '__main__':
    unittest.main()