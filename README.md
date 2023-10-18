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

| Type of Hyperparameter | Explanation                                                              |  
|:-----------------------|:-------------------------------------------------------------------------|
| Float()                | Discrete and continuous values as possible values for the hyperparameter | 
| Integer()              | Only discrete values as possible values for the hyperparameter           | 
| Binary()               | Only binary values as possible values for the hyperparameter             |
| Categorical()          | Only categorical values as possible values for the hyperparameter        |
| Constant()             | hyperparameter where the values does not get changed                     |


Let's define our HyperparameterConfigurationSpace from the example:
```python
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal

cs = HyperparameterConfigurationSpace(
    values={
        "lr": Float("lr", bounds=(0.0001, 0.01), default=0.01, distribution=Normal(0.005, 0.01)),
        "n_layers": Integer("n_layers", bounds=(1, 6), default=3),
        "optimizer_type": Categorical("optimizer_type", choices=["adam", "adamw", "sgd"], default="adam"),
        "number_of_epochs": Constant("number_of_epochs", default=20),
    }
)
cs.get_default_configuration()
```

With the given HyperparameterConfigurationSpace we can now sample Hyperparameters randomly by using the method 
`sample_configuration()`:

```python
samples = cs.sample_configuration(size=10)
```

If you want to get the default values for all hyperparameters, you can use the method `get_default_configuration()`:
```python
default_cfg = cs.get_default_configuration()
```

### Additional Features

Additionally, you can add a random number seed (e. g.: seed=1234) to reproduce the sampling procedure:
```python
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal

cs = HyperparameterConfigurationSpace(
    values={
        "lr": Float("lr", bounds=(0.0001, 0.01), default=0.01, distribution=Normal(0.005, 0.01)),
        "n_layers": Integer("n_layers", bounds=(1, 6), default=3),
        "optimizer_type": Categorical("optimizer_type", choices=["adam", "adamw", "sgd"], default="adam"),
        "number_of_epochs": Constant("number_of_epochs", default=20),
    },
    seed=1234,
)
```

For Float(), Integer() and Constant() it is possible to also create Hyperparameters where the values are matrices 
instead of single values:

```python
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Uniform

cs = HyperparameterConfigurationSpace(
    values={
        "weights1": Float("weights1", bounds=(-1.0, 1.0), default=np.array([[0.0, 0.1], [0.2, 0.3]]) ,distribution=Uniform()),
        "weights2": Integer("weights2", bounds=(1, 6), default=np.array([[1, 2], [3, 4]])),
        "weights3": Constant("weights3", default=np.array([[True, False], [True, True]])),
    }
)
```

### Future Features
The following list defines features, that are currently on work:

- Add Constraints to HyperparameterConfigurationSpace to also add Hierarchical Hyperparameters
- Adjust Binary() and Categorical() to also use values that are matrices instead of single values
- Add support for torch.Tensor to all types of Hyperparameters
- Dynamically adjust shape=... parameter, given to the default value
