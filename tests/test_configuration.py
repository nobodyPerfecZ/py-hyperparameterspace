import unittest
import numpy as np
import yaml

from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class TestHyperparameterConfiguration(unittest.TestCase):
    """
    Tests the class HyperparameterConfiguration.
    """

    def setUp(self) -> None:
        self.cfg = HyperparameterConfiguration(
            values={
                "X1": 0.1,
                "X2": True,
                "X3": 1,
                "X4": np.array(["attribute1", "attribute2"]),
                "X5": -1.2,
            },
        )
        self.cfg2 = HyperparameterConfiguration(
            values={
                "X1": 0.2,
                "X2": False,
                "X3": 3,
                "X4": np.array(["attribute1", "attribute2"]),
                "X5": -1.2,
            },
        )

    def test_save_yaml(self):
        """
        Tests the method save_yaml().
        """
        # Safe the configuration as yaml file
        self.cfg.save_yaml("test_data.yaml")

        # Load the configuration from the yaml file
        cfg = HyperparameterConfiguration.load_yaml("test_data.yaml")

        self.assertTrue(np.array_equal(self.cfg, cfg))

    def test_save_json(self):
        """
        Tests the method save_json().
        """
        # Safe the configuration as json file
        self.cfg.save_json("test_data.json")

        # Load the configuration from the json file
        cfg = HyperparameterConfiguration.load_json("test_data.json")

        self.assertTrue(np.array_equal(self.cfg, cfg))

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


if __name__ == '__main__':
    unittest.main()
