import unittest
import numpy as np

from PyHyperparameterSpace.hp.constant import Constant


class TestConstant(unittest.TestCase):
    """
    Tests the class Constant.
    """

    def setUp(self):
        self.name = "X1"
        self.shape = (1,)
        self.shape2 = (2, 2)
        self.default = "X1"
        self.default2 = np.array([["attr1", "attr2"], ["attr3", "attr4"]])
        self.random = np.random.RandomState()
        self.size = 10
        self.hp = Constant(name=self.name, default=self.default, shape=self.shape)
        self.hp2 = Constant(name=self.name, default=self.default2, shape=self.shape2)

    def test_name(self):
        """
        Tests the property name
        """
        self.assertEqual(self.name, self.hp._name)
        self.assertEqual(self.name, self.hp2._name)

    def test_shape(self):
        """
        Tests the property shape.
        """
        self.assertEqual(self.shape, self.hp._shape)
        self.assertEqual(self.shape2, self.hp2._shape)

    def test_bounds(self):
        """
        Tests the property bounds.
        """
        self.assertIsNone(self.hp._bounds)
        self.assertIsNone(self.hp2._bounds)

    def test_choices(self):
        """
        Tests the property choices.
        """
        self.assertIsNone(self.hp._choices)
        self.assertIsNone(self.hp2._choices)

    def test_default(self):
        """
        Tests the property default.
        """
        self.assertEqual(self.default, self.hp._default)
        self.assertTrue(np.all(self.default2 == self.hp2._default))

    def test_distribution(self):
        """
        Tests the property distribution.
        """
        self.assertIsNone(self.hp._distribution)
        self.assertIsNone(self.hp2._distribution)

    def test_weights(self):
        """
        Tests the property weights.
        """
        self.assertIsNone(self.hp._weights)
        self.assertIsNone(self.hp2._weights)

    def test_lb(self):
        """
        Tests the property lb.
        """
        self.assertIsNone(self.hp._lb)
        self.assertIsNone(self.hp2._lb)

    def test_ub(self):
        """
        Tests the property ub.
        """
        self.assertIsNone(self.hp._ub)
        self.assertIsNone(self.hp2._ub)

    def test_get_name(self):
        """
        Tests the method get_name().
        """
        self.assertEqual(self.name, self.hp.get_name())
        self.assertEqual(self.name, self.hp2.get_name())

    def test_get_default(self):
        """
        Tests the method get_default().
        """
        self.assertEqual(self.default, self.hp.get_default())
        self.assertTrue(np.all(self.default2 == self.hp2.get_default()))

    def test_get_shape(self):
        """
        Tests the method get_shape().
        """
        self.assertEqual(self.shape, self.hp.get_shape())
        self.assertEqual(self.shape2, self.hp2.get_shape())

    def test_sample(self):
        """
        Tests the method sample().
        """
        sample = self.hp.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample))
        self.assertTrue(s == self.default for s in sample)

        sample2 = self.hp2.sample(random=self.random, size=self.size)
        self.assertEqual(self.size, len(sample2))
        self.assertEqual(self.shape2, sample2[0].shape)
        self.assertTrue(np.all(self.default2 == sample2[0]))

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
        self.assertEqual(hash(self.hp), hash(self.hp))
        self.assertNotEqual(hash(self.hp), hash(self.hp2))


if __name__ == '__main__':
    unittest.main()
