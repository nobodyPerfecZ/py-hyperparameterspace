import os
import unittest

import numpy as np
import yaml

from PyHyperparameterSpace.dist.continuous import MatrixNormal, MultivariateNormal, Normal, Uniform
from PyHyperparameterSpace.hp.continuous import Float


class TestFloat(unittest.TestCase):
    """
    Tests the class Float.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.shape2 = (2, 2)
        self.shape3 = (2,)
        self.bounds = (-12.5, 12.5)
        self.default = 0.5
        self.default2 = [[0.5, 0.5], [0.5, 0.5]]
        self.default3 = np.array([[0, 0], [0, 0]])
        self.default4 = [0.5, 1.5]
        self.random = np.random.RandomState()
        self.size = 10
        self.multivariate_normal_distribution = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 100]])
        self.matrix_normal_distribution = MatrixNormal(M=[[0, 0], [0, 0]], U=[[1, 0], [0, 1]], V=[[1, 0], [0, 1]])
        self.normal_distribution = Normal(mean=0.0, std=10.0)
        self.uniform_distribution = Uniform(*self.bounds)

        # Tests with all options given
        self.normal_hp = Float(name=self.name, shape=self.shape, bounds=self.bounds, default=self.default,
                               distribution=self.normal_distribution)
        # Tests with all options given (matrix as values)
        self.normal_hp2 = Float(name=self.name, shape=self.shape2, bounds=self.bounds, default=self.default2,
                                distribution=self.normal_distribution)
        # Tests with default=None
        self.normal_hp3 = Float(name=self.name, shape=self.shape2, bounds=self.bounds, default=None,
                                distribution=self.normal_distribution)
        # Test with shape=None
        self.normal_hp4 = Float(name=self.name, shape=None, bounds=self.bounds, default=self.default2,
                                distribution=self.normal_distribution)
        # Tests with all options given
        self.uniform_hp = Float(name=self.name, shape=self.shape, bounds=self.bounds, default=self.default,
                                distribution=self.uniform_distribution)
        # Tests with all options given (matrix as values)
        self.uniform_hp2 = Float(name=self.name, shape=self.shape2, bounds=self.bounds, default=self.default2,
                                 distribution=self.uniform_distribution)
        # Tests with default=None
        self.uniform_hp3 = Float(name=self.name, shape=self.shape2, bounds=self.bounds, default=None,
                                 distribution=self.uniform_distribution)
        # Tests with shape=None
        self.uniform_hp4 = Float(name=self.name, shape=None, bounds=self.bounds, default=self.default2,
                                 distribution=self.uniform_distribution)

        # Tests with multivariate normal distribution
        self.multivariate_normal_hp = Float(name=self.name, shape=None, bounds=self.bounds, default=self.default4,
                                            distribution=self.multivariate_normal_distribution)

        # Tests with matrix normal distribution
        self.matrix_normal_hp = Float(name=self.name, shape=None, bounds=self.bounds, default=self.default3,
                                      distribution=self.matrix_normal_distribution)

    def test_name(self):
        """
        Tests the property name.
        """
        self.assertEqual(self.name, self.normal_hp._name)
        self.assertEqual(self.name, self.normal_hp2._name)
        self.assertEqual(self.name, self.normal_hp3._name)
        self.assertEqual(self.name, self.normal_hp4._name)
        self.assertEqual(self.name, self.uniform_hp._name)
        self.assertEqual(self.name, self.uniform_hp2._name)
        self.assertEqual(self.name, self.uniform_hp3._name)
        self.assertEqual(self.name, self.uniform_hp4._name)
        self.assertEqual(self.name, self.multivariate_normal_hp._name)
        self.assertEqual(self.name, self.matrix_normal_hp._name)

    def test_shape(self):
        """
        Tests the property shape.
        """
        self.assertEqual(self.shape, self.normal_hp._shape)
        self.assertEqual(self.shape2, self.normal_hp2._shape)
        self.assertEqual(self.shape2, self.normal_hp3._shape)
        self.assertEqual(self.shape2, self.normal_hp4._shape)
        self.assertEqual(self.shape, self.uniform_hp._shape)
        self.assertEqual(self.shape2, self.uniform_hp2._shape)
        self.assertEqual(self.shape2, self.uniform_hp3._shape)
        self.assertEqual(self.shape2, self.uniform_hp4._shape)
        self.assertEqual(self.shape3, self.multivariate_normal_hp._shape)
        self.assertEqual(self.shape2, self.matrix_normal_hp._shape)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        self.assertEqual(self.bounds, self.normal_hp._bounds)
        self.assertEqual(self.bounds, self.normal_hp2._bounds)
        self.assertEqual(self.bounds, self.normal_hp3._bounds)
        self.assertEqual(self.bounds, self.normal_hp4._bounds)
        self.assertEqual(self.bounds, self.uniform_hp._bounds)
        self.assertEqual(self.bounds, self.uniform_hp2._bounds)
        self.assertEqual(self.bounds, self.uniform_hp3._bounds)
        self.assertEqual(self.bounds, self.uniform_hp4._bounds)
        self.assertEqual(self.bounds, self.multivariate_normal_hp._bounds)
        self.assertEqual(self.bounds, self.matrix_normal_hp._bounds)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default, self.normal_hp._default)
        self.assertTrue(np.all(self.default2 == self.normal_hp2._default))
        self.assertTrue(np.all(self.default3 == self.normal_hp3._default))
        self.assertTrue(np.all(self.default2 == self.normal_hp4._default))
        self.assertEqual(self.default, self.uniform_hp._default)
        self.assertTrue(np.all(self.default2 == self.uniform_hp2._default))
        self.assertTrue(np.all(self.default3 == self.uniform_hp3._default))
        self.assertTrue(np.all(self.default2 == self.uniform_hp4._default))
        self.assertTrue(np.all(self.default4 == self.multivariate_normal_hp._default))
        self.assertTrue(np.all(self.default3 == self.matrix_normal_hp._default))

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsInstance(self.normal_hp._distribution, Normal)
        self.assertIsInstance(self.normal_hp2._distribution, Normal)
        self.assertIsInstance(self.normal_hp3._distribution, Normal)
        self.assertIsInstance(self.normal_hp4._distribution, Normal)
        self.assertIsInstance(self.uniform_hp._distribution, Uniform)
        self.assertIsInstance(self.uniform_hp2._distribution, Uniform)
        self.assertIsInstance(self.uniform_hp3._distribution, Uniform)
        self.assertIsInstance(self.uniform_hp4._distribution, Uniform)
        self.assertIsInstance(self.multivariate_normal_hp._distribution, MultivariateNormal)
        self.assertIsInstance(self.matrix_normal_hp._distribution, MatrixNormal)

    def test_lb(self):
        """
        Tests the property lb.
        """
        self.assertEqual(self.bounds[0], self.normal_hp.lb)
        self.assertEqual(self.bounds[0], self.normal_hp2.lb)
        self.assertEqual(self.bounds[0], self.normal_hp3.lb)
        self.assertEqual(self.bounds[0], self.normal_hp4.lb)
        self.assertEqual(self.bounds[0], self.uniform_hp.lb)
        self.assertEqual(self.bounds[0], self.uniform_hp2.lb)
        self.assertEqual(self.bounds[0], self.uniform_hp3.lb)
        self.assertEqual(self.bounds[0], self.uniform_hp4.lb)
        self.assertEqual(self.bounds[0], self.multivariate_normal_hp.lb)
        self.assertEqual(self.bounds[0], self.matrix_normal_hp.lb)

    def test_ub(self):
        """
        Tests the property ub.
        """
        self.assertEqual(self.bounds[1], self.normal_hp.ub)
        self.assertEqual(self.bounds[1], self.normal_hp2.ub)
        self.assertEqual(self.bounds[1], self.normal_hp3.ub)
        self.assertEqual(self.bounds[1], self.normal_hp4.ub)
        self.assertEqual(self.bounds[1], self.uniform_hp.ub)
        self.assertEqual(self.bounds[1], self.uniform_hp2.ub)
        self.assertEqual(self.bounds[1], self.uniform_hp3.ub)
        self.assertEqual(self.bounds[1], self.uniform_hp4.ub)
        self.assertEqual(self.bounds[1], self.multivariate_normal_hp.ub)
        self.assertEqual(self.bounds[1], self.matrix_normal_hp.ub)

    def test_get_name(self):
        """
        Tests the method get_name().
        """
        self.assertEqual(self.name, self.normal_hp.get_name())
        self.assertEqual(self.name, self.normal_hp2.get_name())
        self.assertEqual(self.name, self.normal_hp3.get_name())
        self.assertEqual(self.name, self.normal_hp4.get_name())
        self.assertEqual(self.name, self.uniform_hp.get_name())
        self.assertEqual(self.name, self.uniform_hp2.get_name())
        self.assertEqual(self.name, self.uniform_hp3.get_name())
        self.assertEqual(self.name, self.uniform_hp4.get_name())
        self.assertEqual(self.name, self.multivariate_normal_hp.get_name())
        self.assertEqual(self.name, self.matrix_normal_hp.get_name())

    def test_get_default(self):
        """
        Tests the method get_default().
        """
        self.assertEqual(self.default, self.normal_hp.get_default())
        self.assertTrue(np.all(self.default2 == self.normal_hp2.get_default()))
        self.assertTrue(np.all(self.default3 == self.normal_hp3.get_default()))
        self.assertTrue(np.all(self.default2 == self.normal_hp4.get_default()))
        self.assertEqual(self.default, self.uniform_hp.get_default())
        self.assertTrue(np.all(self.default2 == self.uniform_hp2.get_default()))
        self.assertTrue(np.all(self.default3 == self.uniform_hp3.get_default()))
        self.assertTrue(np.all(self.default2 == self.uniform_hp4.get_default()))
        self.assertTrue(np.all(self.default4 == self.multivariate_normal_hp.get_default()))
        self.assertTrue(np.all(self.default3 == self.matrix_normal_hp.get_default()))

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.normal_hp.get_shape())
        self.assertEqual(self.shape2, self.normal_hp2.get_shape())
        self.assertEqual(self.shape2, self.normal_hp3.get_shape())
        self.assertEqual(self.shape2, self.normal_hp4.get_shape())
        self.assertEqual(self.shape, self.uniform_hp.get_shape())
        self.assertEqual(self.shape2, self.uniform_hp2.get_shape())
        self.assertEqual(self.shape2, self.uniform_hp3.get_shape())
        self.assertEqual(self.shape2, self.uniform_hp4.get_shape())
        self.assertEqual(self.shape3, self.multivariate_normal_hp.get_shape())
        self.assertEqual(self.shape2, self.matrix_normal_hp.get_shape())

    def test_get_bounds(self):
        """
        Tests the method get_bounds().
        """
        self.assertEqual(self.bounds, self.normal_hp.get_bounds())
        self.assertEqual(self.bounds, self.normal_hp2.get_bounds())
        self.assertEqual(self.bounds, self.normal_hp3.get_bounds())
        self.assertEqual(self.bounds, self.normal_hp4.get_bounds())
        self.assertEqual(self.bounds, self.uniform_hp.get_bounds())
        self.assertEqual(self.bounds, self.uniform_hp2.get_bounds())
        self.assertEqual(self.bounds, self.uniform_hp3.get_bounds())
        self.assertEqual(self.bounds, self.uniform_hp4.get_bounds())
        self.assertEqual(self.bounds, self.multivariate_normal_hp.get_bounds())
        self.assertEqual(self.bounds, self.matrix_normal_hp.get_bounds())

    def test_get_distribution(self):
        """
        Tests the method get_distribution().
        """
        self.assertIsInstance(self.normal_hp.get_distribution(), Normal)
        self.assertIsInstance(self.normal_hp2.get_distribution(), Normal)
        self.assertIsInstance(self.normal_hp3.get_distribution(), Normal)
        self.assertIsInstance(self.normal_hp4.get_distribution(), Normal)
        self.assertIsInstance(self.uniform_hp.get_distribution(), Uniform)
        self.assertIsInstance(self.uniform_hp2.get_distribution(), Uniform)
        self.assertIsInstance(self.uniform_hp3.get_distribution(), Uniform)
        self.assertIsInstance(self.uniform_hp4.get_distribution(), Uniform)
        self.assertIsInstance(self.multivariate_normal_hp.get_distribution(), MultivariateNormal)
        self.assertIsInstance(self.matrix_normal_hp.get_distribution(), MatrixNormal)

    def test_change_distribution(self):
        """
        Tests the method change_distribution().
        """
        new_mean = 0.5
        new_std = 1.5
        self.normal_hp.change_distribution(mean=new_mean, std=new_std)

        new_lb = -0.5
        new_ub = 0.5
        self.uniform_hp.change_distribution(lb=new_lb, ub=new_ub)

        distribution = self.normal_hp.get_distribution()
        self.assertEqual(new_mean, distribution.mean)
        self.assertEqual(new_std, distribution.std)

        distribution = self.uniform_hp.get_distribution()
        self.assertEqual(new_lb, distribution.lb)
        self.assertEqual(new_ub, distribution.ub)

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

        normal_sample2 = self.normal_hp2.sample(random=self.random, size=self.size)
        uniform_sample2 = self.uniform_hp2.sample(random=self.random, size=self.size)

        self.assertEqual(self.size, len(normal_sample2))
        self.assertEqual(self.size, len(uniform_sample2))
        self.assertEqual(self.shape2, normal_sample2[0].shape)
        self.assertEqual(self.shape2, uniform_sample2[0].shape)
        self.assertTrue(np.all((self.bounds[0] <= normal_sample2) & (normal_sample2 < self.bounds[1])))
        self.assertTrue(np.all((self.bounds[0] <= uniform_sample2) & (uniform_sample2 < self.bounds[1])))

        multivariate_normal_sample = self.multivariate_normal_hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(multivariate_normal_sample))
        self.assertEqual(self.shape3, multivariate_normal_sample[0].shape)
        self.assertTrue(
            np.all((self.bounds[0] <= multivariate_normal_sample) & (multivariate_normal_sample < self.bounds[1])))

        matrix_normal_sample = self.matrix_normal_hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(matrix_normal_sample))
        self.assertEqual(self.shape2,  matrix_normal_sample[0].shape)
        self.assertTrue(
            np.all((self.bounds[0] <= matrix_normal_sample) & (matrix_normal_sample < self.bounds[1])))

    def test_valid_configuration(self):
        """
        Tests the method valid_configuration().
        """
        self.assertTrue(self.normal_hp.valid_configuration(self.default))
        self.assertFalse(self.normal_hp.valid_configuration(self.default3))

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.normal_hp, self.normal_hp)
        self.assertNotEqual(self.normal_hp, self.normal_hp2)
        self.assertNotEqual(self.normal_hp, self.normal_hp3)
        self.assertNotEqual(self.normal_hp, self.normal_hp4)
        self.assertEqual(self.uniform_hp, self.uniform_hp)
        self.assertNotEqual(self.uniform_hp, self.uniform_hp2)
        self.assertNotEqual(self.uniform_hp, self.uniform_hp3)
        self.assertNotEqual(self.uniform_hp, self.uniform_hp4)
        self.assertNotEqual(self.uniform_hp, self.multivariate_normal_hp)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertEqual(hash(self.normal_hp), hash(self.normal_hp))
        self.assertNotEqual(hash(self.normal_hp), hash(self.normal_hp2))
        self.assertNotEqual(hash(self.normal_hp), hash(self.normal_hp3))
        self.assertNotEqual(hash(self.normal_hp), hash(self.normal_hp4))
        self.assertEqual(hash(self.uniform_hp), hash(self.uniform_hp))
        self.assertNotEqual(hash(self.uniform_hp), hash(self.uniform_hp2))
        self.assertNotEqual(hash(self.uniform_hp), hash(self.uniform_hp3))
        self.assertNotEqual(hash(self.uniform_hp), hash(self.uniform_hp4))
        self.assertNotEqual(hash(self.uniform_hp), hash(self.multivariate_normal_hp))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.normal_hp, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            hp = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(hp, self.normal_hp)

        # Delete the yaml file
        os.remove("test_data.yaml")


if __name__ == '__main__':
    unittest.main()
