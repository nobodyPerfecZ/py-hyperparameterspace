# PyHyperparameterSpace
PyHyperparameterSpace is a simple Python Framework for managing your Hyperparameters for Hyperparameter Optimization 
(HPO) Tasks.
You can find more information about HPO Tasks [here](https://en.wikipedia.org/wiki/Hyperparameter_optimization).

### Managing Hyperparameters with PyHyperparameterSpace
In the following, we want to manage the following hyperparameters for our HPO task:

| Hyperparameter   | Value Ranges               | Default | Explanation                   |
|:-----------------|:---------------------------|:--------|:------------------------------|
| lr               | [0.0001; 0.01) (exclusive) | 0.01    | learning rate of an optimizer |
| n_layers         | [1, 6) (exclusive)         | 3       | number of layers to use       |
| optimizer_type   | ["adam", "adamw", "sgd"]   | "adam"  | type of the used optimizer    |
| number_of_epochs | -                          | 20      | number of training epochs     |

PyHyperparameterSpace classifies each Hyperparameter in the following categories:

| Type of Hyperparameter | Explanation                             | Example                                                 |
|:-----------------------|:----------------------------------------|:--------------------------------------------------------|
| Float                  | Values are in the real number domain    | learning rate of an optimizer (lr)                      |
| Integer                | Values are in the natural number domain | number of layers (n_layers)                             |
| Categorical            | Values are in a set of choices          | type of the optimizer to use (optimizer_type)           |
| Constant               | Single unchanged value                  | number of epochs to train a NN model (number_of_epochs) |


Let's define our HyperparameterConfigurationSpace from the example:
```python
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal

cs = HyperparameterConfigurationSpace(
    values={
        "lr": Float("lr", bounds=(1e-4, 1e-2), default=1e-3, distribution=Normal(0.005, 0.01)),
        "n_layers": Integer("n_layers", bounds=(1, 6), default=3),
        "optimizer_type": Categorical("optimizer_type", choices=["adam", "adamw", "sgd"], default="adam"),
        "number_of_epochs": Constant("number_of_epochs", default=20),
    }
)
```

With the given HyperparameterConfigurationSpace, we can now sample hyperparameters randomly by using the method 
`sample_configuration()`:

```python
samples = cs.sample_configuration(size=10)
```

If you want to get the default values for all hyperparameters, you can use the method `get_default_configuration()`:
```python
default_cfg = cs.get_default_configuration()
```

### Additional Features

Additionally, you can add a random number seed to reproduce the sampling procedure:
```python
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal

cs = HyperparameterConfigurationSpace(
    values={
        "lr": Float("lr", bounds=(1e-4, 1e-2), default=1e-3, distribution=Normal(0.005, 0.01)),
        "n_layers": Integer("n_layers", bounds=(1, 6), default=3),
        "optimizer_type": Categorical("optimizer_type", choices=["adam", "adamw", "sgd"], default="adam"),
        "number_of_epochs": Constant("number_of_epochs", default=20),
    },
    seed=1234,
)
```

It is also possible to create hyperparameters with multidimensional values instead of single dimensional values:

```python
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Uniform

cs = HyperparameterConfigurationSpace(
    values={
        "hp1": Float("hp1", bounds=(-1.0, 1.0), default=[[0.0, 0.1], [0.2, 0.3]], distribution=Uniform()),
        "hp2": Integer("hp2", bounds=(1, 6), default=[[1, 2], [3, 4]]),
        "hp3": Categorical("hp3", choices=[["adam", "use_weight_decay"], ["adamw", "use_weight_decay"], ["sgd", "no_weight_decay"]]),
        "hp4": Constant("hp4", default=[[True, False], [True, True]]),
    }
)
```

### Future Features
The following list defines features, that are currently on work:

* [ ] Add Constraints to HyperparameterConfigurationSpace to also add Hierarchical Hyperparameters
* [x] Implement Matrix normal distribution as sampling option for Float Hyperparameters
* [x] Implement saving functions of HyperparameterConfiguration in JSON and YML format
* [x] Remove Binary() class, because it is very similar to categorical
* [x] (For Binary() not possible due to the nature of binary values) Adjust Binary() and Categorical() to also use values that are matrices instead of single values
* [x] (Discarded due to lower performance) Add support for torch.Tensor to all types of Hyperparameters
* [x] Dynamically adjust shape=... parameter, given to the default value
