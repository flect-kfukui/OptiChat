"""Type definitions for the OptiChat system."""

from typing import Any, Dict, Set

import pyomo.environ as pe
from typing_extensions import NotRequired, TypedDict


class TeamConversationMessage(TypedDict):
    """
    Type definition for team conversation messages.

    Attributes
    ----------
    agent_name : str
        The name of the agent that generated the message.
    agent_response : str
        The response content from the agent.
    """

    agent_name: str
    agent_response: str


class DecisionDict(TypedDict):
    """
    Type definition for coordinator decision dictionaries.

    Used by the Coordinator agent to specify which agent should work next
    and what task they should perform.

    Attributes
    ----------
    agent_name : str
        The name of the agent to call next (e.g., "Engineer", "Explainer").
    task : str
        The task for the agent to carry out. Can be "DONE" to indicate completion.
    """

    agent_name: str
    task: str


# TypedDict definitions for model dictionary structures
#
# These type definitions provide structure and documentation for the complex
# nested dictionaries used throughout this module to represent optimization models.
#
# Key Structure Overview:
# - ModelsContainer: Top-level container with "model_representation" and "model_1" keys
# - ModelDictWithPyomo: Individual model dict containing Pyomo objects
#   (not serializable)
# - ModelDictSerializable: Model dict without Pyomo objects (JSON serializable)
# - ComponentsContainer: Nested structure containing sets, parameters, variables, etc.
#
# Note: Some dictionary keys use spaces (e.g., "model status") which limits full
# TypedDict compatibility. These types serve primarily as documentation and partial
# type checking. For complete type safety, consider refactoring to use underscore keys.


class SetComponent(TypedDict):
    """Dictionary structure for optimization model sets."""

    name: str
    is_indexed: bool
    description: str


class ParameterComponent(TypedDict):
    """Dictionary structure for optimization model parameters."""

    name: str
    is_indexed: bool
    index_set: Any  # Pyomo index set object or None
    is_RHS: bool
    is_mutable: bool
    cons_in: Set[str]
    description: str


class ParameterComponentSerializable(TypedDict):
    """Serializable dictionary structure for model parameters (no Pyomo objects)."""

    name: str
    is_indexed: bool
    is_RHS: bool
    is_mutable: bool
    cons_in: Set[str]
    description: str


class VariableComponent(TypedDict):
    """Dictionary structure for optimization model variables."""

    name: str
    is_indexed: bool
    index_set: Any  # Pyomo index set object or None
    cons_in: Set[str]
    description: str


class VariableComponentSerializable(TypedDict):
    """Serializable dictionary structure for model variables (no Pyomo objects)."""

    name: str
    is_indexed: bool
    cons_in: Set[str]
    description: str


class ConstraintComponent(TypedDict):
    """Dictionary structure for optimization model constraints."""

    name: str
    is_indexed: bool
    index_set: Any  # Pyomo index set object or None
    params_in: Set[str]
    vars_in: Set[str]
    description: str


class ConstraintComponentSerializable(TypedDict):
    """Serializable dictionary structure for model constraints (no Pyomo objects)."""

    name: str
    is_indexed: bool
    params_in: Set[str]
    vars_in: Set[str]
    description: str


class ObjectiveComponent(TypedDict):
    """Dictionary structure for optimization model objectives."""

    name: str
    sense: str
    optimal_value: Any  # Can be numeric value or string for infeasible cases
    is_indexed: bool
    description: str


class ComponentsContainer(TypedDict):
    """Dictionary structure for all model components."""

    sets: Dict[str, SetComponent]
    parameters: Dict[str, ParameterComponent]
    variables: Dict[str, VariableComponent]
    constraints: Dict[str, ConstraintComponent]
    objective: Dict[str, ObjectiveComponent]


class ComponentsContainerSerializable(TypedDict):
    """Dictionary structure for model components (serializable, no Pyomo objects)."""

    sets: Dict[str, SetComponent]  # Sets don't have index_set field
    parameters: Dict[str, ParameterComponentSerializable]
    variables: Dict[str, VariableComponentSerializable]
    constraints: Dict[str, ConstraintComponentSerializable]
    objective: Dict[str, ObjectiveComponent]  # Objectives don't have index_set


class IISConstraint(TypedDict):
    """Dictionary structure for IIS constraint information."""

    params_in: Set[str]
    vars_in: Set[str]


class ModelDictWithPyomo(TypedDict):
    """
    Dictionary structure for a complete optimization model with Pyomo objects.

    Uses underscore keys for proper TypedDict support:
    - model_class: Pyomo ConcreteModel object
    - model_status: Solver termination condition
    - model_type: Problem type ("LP", "IP", etc.)
    - model_description: Model description string
    - components: Nested dict with sets, parameters, variables, constraints, objective
    - code: Source code string
    - iis: Optional IIS constraint information
    - iis_description: Optional IIS description string
    """

    # Required fields from pyomo2json
    model_class: pe.ConcreteModel
    model_status: str
    model_type: str
    model_description: Any
    components: ComponentsContainer

    # Added in initial_loading
    code: str

    # Optional fields added conditionally
    iis: NotRequired[Dict[str, IISConstraint]]
    iis_description: NotRequired[str]


class ModelDictSerializable(TypedDict):
    """
    Dictionary structure for a serializable optimization model (no Pyomo objects).

    Same as ModelDictWithPyomo but excludes model_class and uses serializable
    components that don't contain index_set fields with Pyomo objects.

    Uses underscore keys for proper TypedDict support:
    - model_status: Solver termination condition
    - model_type: Problem type ("LP", "IP", etc.)
    - model_description: Model description string
    - components: Serializable nested dict with model components
    - code: Source code string
    - iis: Optional IIS constraint information
    - iis_description: Optional IIS description string
    """

    # Required fields (no model_class since it's not serializable)
    model_status: str
    model_type: str
    model_description: Any
    components: ComponentsContainerSerializable

    # Added in initial_loading
    code: str

    # Optional fields added conditionally
    iis: NotRequired[Dict[str, IISConstraint]]
    iis_description: NotRequired[str]


# For the main models container
ModelsContainer = Dict[str, ModelDictWithPyomo]
