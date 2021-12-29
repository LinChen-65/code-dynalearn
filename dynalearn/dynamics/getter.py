from .stochastic_epidemics import (
    SIS,
    SIR,
    PlanckSIS,
    AsymmetricSISSIS,
)
from .deterministic_epidemics import DSIR, IncSIR
from .trainable import (
    GNNSEDynamics,
    GNNDEDynamics,
    GNNIncidenceDynamics,
    VARDynamics,
    KapoorDynamics,
)

__dynamics__ = {
    "SIS": SIS,
    "SIR": SIR,
    "DSIR": DSIR,
    "IncSIR": IncSIR,
    "PlanckSIS": PlanckSIS,
    "AsymmetricSISSIS": AsymmetricSISSIS,
    "GNNSEDynamics": GNNSEDynamics,
    "TrainableStochasticEpidemics": GNNSEDynamics,
    "GNNDEDynamics": GNNDEDynamics,
    "GNNIncidenceDynamics": GNNIncidenceDynamics,
    "VARDynamics": VARDynamics,
    "KapoorDynamics": KapoorDynamics,
}


def get(config):
    name = config.name
    if name in __dynamics__:
        return __dynamics__[name](config) #回到experiments/experiment.py(48)__init__()
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__dynamics__.keys())}"
        )
