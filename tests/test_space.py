import unittest
import numpy as np
import yaml

from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.space import HyperparameterConfiguration
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal, Uniform
from PyHyperparameterSpace.dist.categorical import Choice


class TestHyperparameterConfigurationSpace(unittest.TestCase):
    """
    Tests the class HyperparameterConfigurationSpace.
    """

    def setUp(self) -> None:
        self.cs = HyperparameterConfigurationSpace(
            values={
                "X1": Float("X1", bounds=(-10.5, 10.5), default=2.25, shape=(1,), distribution=Normal(2.25, 2.5)),
                "X2": Categorical("X2", choices=[True, False], default=True),
                "X3": Integer("X3", bounds=(-10, 10), default=-5, shape=(1,)),
                "X4": Categorical("X4", choices=["attr1", "attr2", "attr3"], default="attr1", distribution=Choice([0.3, 0.4, 0.3])),
                "X5": Constant("X5", default="X_Const", shape=(1,)),
            },
            seed=0,
        )
        self.new_hps = [
            Integer("X6", bounds=(-3, 3), default=0, shape=(1,)),
            Categorical("X7", choices=[True, False], default=True),
        ]
        self.cs2 = HyperparameterConfigurationSpace(
            values={
                "X1": Float("X1", bounds=(-10.5, 10.5), default=2.25, shape=(1,), distribution=Normal(2.25, 2.5)),
                "X2": Categorical("X2", choices=[True, False], default=False),
                "X3": Integer("X3", bounds=(-10, 10), default=0, shape=(1,)),
                "X4": Categorical("X4", choices=["attr1", "attr2", "attr3"], default="attr3", distribution=Choice([0.4, 0.3, 0.3])),
                "X5": Constant("X5", default=np.array(["X_Const", "X_Const"]), shape=(2,)),
            },
            seed=1,
        )

        self.size = 10

    def test_add_hyperparameter(self):
        """
        Tests the method add_hyperparameter().
        """
        self.cs.add_hyperparameter(self.new_hps[0])

        self.assertEqual(6, len(self.cs))
        self.assertIn("X6", self.cs)
        self.assertEqual(self.new_hps[0], self.cs["X6"])

    def test_add_hyperparameters(self):
        """
        Tests the method add_hyperparameters().
        """
        self.cs.add_hyperparameters(self.new_hps)

        self.assertEqual(7, len(self.cs))
        self.assertIn("X6", self.cs)
        self.assertIn("X7", self.cs)
        self.assertEqual(self.new_hps[0], self.cs["X6"])
        self.assertEqual(self.new_hps[1], self.cs["X7"])

    def test_sample_configuration(self):
        """
        Tests the method sample_configuration().
        """
        configs = self.cs.sample_configuration()
        configs2 = self.cs.sample_configuration(size=self.size)

        self.assertIsInstance(configs, HyperparameterConfiguration)

        self.assertIsInstance(configs2, list)
        self.assertEqual(self.size, len(configs2))

    def test_get_default_configuration(self):
        """
        Tests the method get_default_configuration().
        """
        configs = self.cs.get_default_configuration()
        configs2 = self.cs.get_default_configuration(size=10)

        self.assertIsInstance(configs, HyperparameterConfiguration)
        self.assertEqual(2.25, configs["X1"])
        self.assertEqual(True, configs["X2"])
        self.assertEqual(-5, configs["X3"])
        self.assertEqual("attr1", configs["X4"])
        self.assertEqual("X_Const", configs["X5"])

        self.assertIsInstance(configs2, list)
        self.assertEqual(self.size, len(configs2))
        self.assertTrue(all(configs == cfg for cfg in configs2))

    def test_contains(self):
        """
        Tests the magic function __contains__.
        """
        self.assertIn("X1", self.cs)
        self.assertIn("X3", self.cs)
        self.assertNotIn("X", self.cs)

    def test_getitem(self):
        """
        Tests the magic function __getitem__.
        """
        self.assertIsInstance(self.cs["X1"], Float)
        self.assertIsInstance(self.cs["X3"], Integer)
        with self.assertRaises(KeyError):
            self.cs["X"]

    def test_setitem(self):
        """
        Tests the magic function __setitem__.
        """
        hp = Float("X1", bounds=(-11, 11), default=2.25, shape=(1,), distribution=Normal(2.25, 2.5))
        self.cs["X1"] = hp

        self.assertEqual(hp, self.cs["X1"])

    def test_len(self):
        """
        Tests the magic function __len__.
        """
        self.assertEqual(5, len(self.cs))

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertNotEqual(self.cs, self.cs2)

    def test_hash(self):
        """
        Tests the magic function __hash__.
        """
        self.assertNotEqual(hash(self.cs), hash(self.cs2))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.cs, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            cs = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(cs, self.cs)


if __name__ == '__main__':
    unittest.main()
