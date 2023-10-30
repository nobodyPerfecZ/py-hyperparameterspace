import unittest
import numpy as np
import yaml

from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class TestHyperparameterConfiguration(unittest.TestCase):
    """
    Tests the class HyperparameterConfiguration.
    """

    def setUp(self) -> None:
        self.cs = HyperparameterConfigurationSpace(
            values={
                "X1": Float("X1", bounds=(-10.5, 10.5), default=2.25, shape=(1,)),
                "X2": Categorical("X2", choices=[True, False], default=True),
                "X3": Integer("X3", bounds=(-10, 10), default=-5, shape=(1,)),
                "X4": Categorical("X4", choices=["attr1", "attr2", "attr3"], default="attr1", weights=[0.3, 0.4, 0.3]),
                "X5": Constant("X5", default="X_Const", shape=(1,)),
            },
            seed=0,
        )
        self.cfg = HyperparameterConfiguration(
            cs=self.cs,
            values={
                "X1": 0.1,
                "X2": True,
                "X3": 1,
                "X4": "attr1",
                "X5": "X_Const",
            },
        )
        self.cfg2 = HyperparameterConfiguration(
            cs=self.cs,
            values={
                "X1": 0.2,
                "X2": False,
                "X3": 3,
                "X4": "attr2",
                "X5": "X_Const",
            },
        )

    def test_contains(self):
        """
        Tests the magic function __contains__.
        """
        self.assertIn("X1", self.cfg)
        self.assertIn("X3", self.cfg)
        self.assertNotIn("X", self.cfg)

    def test_getitem(self):
        """
        Tests the magic function __getitem__.
        """
        self.assertEqual(0.1, self.cfg["X1"])
        self.assertEqual(1, self.cfg["X3"])
        with self.assertRaises(KeyError):
            test = self.cfg["X"]

    def test_setitem(self):
        """
        Tests the magic function __setitem__.
        """
        self.cfg["X"] = 0.4
        self.assertEqual(0.4, self.cfg["X"])

    def test_len(self):
        """
        Tests the magic function __len__.
        """
        self.assertEqual(5, len(self.cfg))

    def test_eq(self):
        """
        Tests the magic function __eq__.
        """
        self.assertEqual(self.cfg, self.cfg)
        self.assertNotEqual(self.cfg, self.cfg2)

    def test_hash(self):
        """
        Tests the function __hash__.
        """
        self.assertNotEqual(hash(self.cfg), hash(self.cfg2))

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.cfg, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            cfg = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(cfg, self.cfg)


if __name__ == '__main__':
    unittest.main()
