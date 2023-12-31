import os
import unittest

import numpy as np
import yaml

from PyHyperparameterSpace.dist.categorical import Choice
from PyHyperparameterSpace.hp.categorical import Categorical


class TestCategorical(unittest.TestCase):
    """
    Tests the class Categorical.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.shape5 = (2, 2)
        self.choices = ["X1", "X2", "X3", "X4", "X5"]
        self.choices5 = [np.array([["X1", "X2"], ["X3", "X4"]]), np.array([["X2", "X3"], ["X4", "X1"]]),
                         np.array([["X3", "X4"], ["X1", "X2"]])]
        self.default_X1 = "X1"
        self.default_X3 = "X3"
        self.default_X5 = "X5"
        self.default5 = np.array([["X2", "X3"], ["X4", "X1"]])

        self.weights = [0.1, 0.1, 0.2, 0.1, 0.5]
        self.new_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.weights_uniform = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.weights5 = [0.3, 0.4, 0.3]

        self.distribution = Choice(self.weights)
        self.distribution2 = Choice(self.weights5)
        self.random = np.random.RandomState()
        self.size = 10

        # Tests with all options given
        self.hp = Categorical(name=self.name, choices=self.choices, default=self.default_X3,
                              distribution=self.distribution)
        # Tests with default=None
        self.hp2 = Categorical(name=self.name, choices=self.choices, default=None,
                               distribution=self.distribution)
        # Tests with distribution=None
        self.hp3 = Categorical(name=self.name, choices=self.choices, default=self.default_X1, distribution=None)
        # Tests with default=None and distribution=None
        self.hp4 = Categorical(name=self.name, choices=self.choices, default=None, distribution=None)
        # Test with default=None, shape=None, distribution=None
        self.hp5 = Categorical(name=self.name, choices=self.choices, default=None, distribution=None, shape=None)
        # Test with all options
        self.hp6 = Categorical(name=self.name, choices=self.choices5, default=self.default5,
                               distribution=self.distribution2, shape=self.shape5)
        # Test with no shape and default given
        self.hp7 = Categorical(name=self.name, choices=self.choices5, default=None, distribution=self.distribution2,
                               shape=None)

    def test_name(self):
        """
        Tests the property name
        """
        self.assertEqual(self.name, self.hp._name)
        self.assertEqual(self.name, self.hp2._name)
        self.assertEqual(self.name, self.hp3._name)
        self.assertEqual(self.name, self.hp4._name)
        self.assertEqual(self.name, self.hp5._name)
        self.assertEqual(self.name, self.hp6._name)
        self.assertEqual(self.name, self.hp7._name)

    def test_shape(self):
        """
        Tests the property shape.
        """
        self.assertEqual(self.shape, self.hp._shape)
        self.assertEqual(self.shape, self.hp2._shape)
        self.assertEqual(self.shape, self.hp3._shape)
        self.assertEqual(self.shape, self.hp4._shape)
        self.assertEqual(self.shape, self.hp5._shape)
        self.assertEqual(self.shape5, self.hp6._shape)
        self.assertEqual(self.shape5, self.hp7._shape)

    def test_choices(self):
        """
        Tests the property choices.
        """
        self.assertTrue(np.array_equal(self.choices, self.hp._choices))
        self.assertTrue(np.array_equal(self.choices, self.hp2._choices))
        self.assertTrue(np.array_equal(self.choices, self.hp3._choices))
        self.assertTrue(np.array_equal(self.choices, self.hp4._choices))
        self.assertTrue(np.array_equal(self.choices, self.hp5._choices))
        self.assertTrue(np.array_equal(self.choices5, self.hp6._choices))
        self.assertTrue(np.array_equal(self.choices5, self.hp7._choices))

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default_X3, self.hp._default)
        self.assertEqual(self.default_X5, self.hp2._default)
        self.assertEqual(self.default_X1, self.hp3._default)
        self.assertEqual(self.default_X1, self.hp4._default)
        self.assertEqual(self.default_X1, self.hp5._default)
        self.assertTrue(np.all(self.default5 == self.hp6._default))
        self.assertTrue(np.all(self.default5 == self.hp7._default))

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsInstance(self.hp._distribution, Choice)
        self.assertIsInstance(self.hp2._distribution, Choice)
        self.assertIsInstance(self.hp3._distribution, Choice)
        self.assertIsInstance(self.hp4._distribution, Choice)
        self.assertIsInstance(self.hp5._distribution, Choice)
        self.assertIsInstance(self.hp6._distribution, Choice)
        self.assertIsInstance(self.hp7._distribution, Choice)

    def test_get_name(self):
        """
        Tests the method get_name().
        """
        self.assertEqual(self.name, self.hp.get_name())
        self.assertEqual(self.name, self.hp2.get_name())
        self.assertEqual(self.name, self.hp3.get_name())
        self.assertEqual(self.name, self.hp4.get_name())
        self.assertEqual(self.name, self.hp5.get_name())
        self.assertEqual(self.name, self.hp6.get_name())
        self.assertEqual(self.name, self.hp7.get_name())

    def test_get_default(self):
        """
        Tests the method get_default().
        """
        self.assertEqual(self.default_X3, self.hp.get_default())
        self.assertEqual(self.default_X5, self.hp2.get_default())
        self.assertEqual(self.default_X1, self.hp3.get_default())
        self.assertEqual(self.default_X1, self.hp4.get_default())
        self.assertEqual(self.default_X1, self.hp5.get_default())
        self.assertTrue(np.all(self.default5 == self.hp6.get_default()))
        self.assertTrue(np.all(self.default5 == self.hp7.get_default()))

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.hp.get_shape())
        self.assertEqual(self.shape, self.hp2.get_shape())
        self.assertEqual(self.shape, self.hp3.get_shape())
        self.assertEqual(self.shape, self.hp4.get_shape())
        self.assertEqual(self.shape, self.hp5.get_shape())
        self.assertEqual(self.shape5, self.hp6.get_shape())
        self.assertEqual(self.shape5, self.hp7.get_shape())

    def test_get_choices(self):
        """
        Tests the method get_choices().
        """
        self.assertTrue(np.array_equal(self.choices, self.hp.get_choices()))
        self.assertTrue(np.array_equal(self.choices, self.hp2.get_choices()))
        self.assertTrue(np.array_equal(self.choices, self.hp3.get_choices()))
        self.assertTrue(np.array_equal(self.choices, self.hp4.get_choices()))
        self.assertTrue(np.array_equal(self.choices, self.hp5.get_choices()))
        self.assertTrue(np.array_equal(self.choices5, self.hp6.get_choices()))
        self.assertTrue(np.array_equal(self.choices5, self.hp7.get_choices()))

    def test_get_distribution(self):
        """
        Tests the method get_distribution().
        """
        self.assertIsInstance(self.hp.get_distribution(), Choice)
        self.assertIsInstance(self.hp2.get_distribution(), Choice)
        self.assertIsInstance(self.hp3.get_distribution(), Choice)
        self.assertIsInstance(self.hp4.get_distribution(), Choice)
        self.assertIsInstance(self.hp5.get_distribution(), Choice)
        self.assertIsInstance(self.hp6.get_distribution(), Choice)
        self.assertIsInstance(self.hp7.get_distribution(), Choice)

    def test_change_distribution(self):
        """
        Tests the method _change_distribution().
        """
        self.hp.change_distribution(weights=self.new_weights)
        self.assertTrue(np.array_equal(self.new_weights, self.hp._distribution.weights))

    def test_sample(self):
        """
        Tests the method sample().
        """
        sample = self.hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s in self.choices for s in sample)

        sample2 = self.hp2.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample2))
        self.assertTrue(s in self.choices for s in sample2)

        sample3 = self.hp3.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample3))
        self.assertTrue(s in self.choices for s in sample3)

        sample4 = self.hp4.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample4))
        self.assertTrue(s in self.choices for s in sample4)

        sample5 = self.hp5.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample5))
        self.assertTrue(s in self.choices for s in sample5)

        sample6 = self.hp6.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample6))
        self.assertTrue(s in self.choices for s in sample6)

        sample7 = self.hp6.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample7))
        self.assertTrue(s in self.choices for s in sample7)

    def test_valid_configuration(self):
        """
        Tests the method valid_configuration().
        """
        self.assertTrue(self.hp.valid_configuration(self.default_X1))
        self.assertFalse(self.hp.valid_configuration(self.default5))

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.hp, self.hp)
        self.assertNotEqual(self.hp, self.hp2)
        self.assertNotEqual(self.hp, self.hp3)
        self.assertNotEqual(self.hp, self.hp4)
        self.assertNotEqual(self.hp, self.hp5)
        self.assertNotEqual(self.hp, self.hp6)
        self.assertNotEqual(self.hp, self.hp7)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertEqual(hash(self.hp), hash(self.hp))
        self.assertNotEqual(hash(self.hp), hash(self.hp2))
        self.assertNotEqual(hash(self.hp), hash(self.hp3))
        self.assertNotEqual(hash(self.hp), hash(self.hp4))
        self.assertNotEqual(hash(self.hp), hash(self.hp5))
        self.assertNotEqual(hash(self.hp), hash(self.hp6))
        self.assertNotEqual(hash(self.hp), hash(self.hp7))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.hp, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            hp = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(hp, self.hp)

        # Delete the yaml file
        os.remove("test_data.yaml")


if __name__ == '__main__':
    unittest.main()
