from typing import TypeVar

from ..instance import InstanceBase
from ..population import PopulationBase


Instance = TypeVar("Instance", bound=InstanceBase)
Population = TypeVar("Population", bound=PopulationBase)
