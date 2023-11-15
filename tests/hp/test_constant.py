import os
import unittest

import numpy as np
import yaml

from PyHyperparameterSpace.hp.constant import Constant


class TestConstant(unittest.TestCase):
    """
    Tests the class Constant.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.shape3 = (2, 2)
        self.default = "X1"
        self.default3 = np.array([["attr1", "attr2"], ["attr3", "attr4"]])
        self.random = np.random.RandomState()
        self.size = 10
        # Test with all options are given
        self.hp = Constant(name=self.name, default=self.default, shape=self.shape)
        # Test with shape=None
        self.hp2 = Constant(name=self.name, default=self.default, shape=None)
        # Test with all options are given (where values are matrices)
        self.hp3 = Constant(name=self.name, default=self.default3, shape=self.shape3)
        # Test with shape=None
        self.hp4 = Constant(name=self.name, default=self.default3, shape=None)

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
        self.assertEqual(self.shape3, self.hp3._shape)
        self.assertEqual(self.shape3, self.hp4._shape)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default, self.hp._default)
        self.assertEqual(self.default, self.hp2._default)
        self.assertTrue(np.all(self.default3 == self.hp3._default))
        self.assertTrue(np.all(self.default3 == self.hp4._default))

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
        self.assertEqual(self.default, self.hp2.get_default())
        self.assertTrue(np.all(self.default3 == self.hp3.get_default()))
        self.assertTrue(np.all(self.default3 == self.hp4.get_default()))

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.hp.get_shape())
        self.assertEqual(self.shape, self.hp2.get_shape())
        self.assertEqual(self.shape3, self.hp3.get_shape())
        self.assertEqual(self.shape3, self.hp4.get_shape())

    def test_sample(self):
        """
        Tests the method sample().
        """
        sample = self.hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s == self.default for s in sample)

        sample2 = self.hp2.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample2))
        self.assertTrue(s == self.default for s in sample2)

        sample3 = self.hp3.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample3))
        self.assertEqual(self.shape3, sample3[0].shape)
        self.assertTrue(np.all(self.default3 == sample3[0]))

        sample4 = self.hp4.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample4))
        self.assertEqual(self.shape3, sample4[0].shape)
        self.assertTrue(np.all(self.default3 == sample4[0]))

    def test_valid_configuration(self):
        """
        Tests the method valid_configuration().
        """
        self.assertTrue(self.hp.valid_configuration(self.default))
        self.assertFalse(self.hp.valid_configuration(self.default3))

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.hp, self.hp)
        self.assertEqual(self.hp, self.hp2)
        self.assertNotEqual(self.hp, self.hp3)
        self.assertNotEqual(self.hp, self.hp4)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertEqual(hash(self.hp), hash(self.hp))
        self.assertEqual(hash(self.hp), hash(self.hp2))
        self.assertNotEqual(hash(self.hp), hash(self.hp3))
        self.assertNotEqual(hash(self.hp), hash(self.hp4))

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
