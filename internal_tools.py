import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import pyomo.environ as pe
from loguru import logger
from pyomo.core.base.constraint import ConstraintData, IndexedConstraint
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import (
    clone_expression,
    identify_mutable_parameters,
    replace_expressions,
)
from pyomo.opt import SolverFactory, TerminationCondition

from extractor import pyomo2json

# OptiChat types
from optichat_types import (
    IndexGuidanceResult,
    ModelDictWithPyomo,
    ModelsContainer,
    QueriedComponent,
    SyntaxGuidanceInternalResult,
)


def fnArgsDecoder(queried_components: list[QueriedComponent]) -> list[QueriedComponent]:
    """
    Decode function arguments by converting string representations to appropriate types.

    Parameters
    ----------
    queried_components : list[QueriedComponent]
        List of component dictionaries containing query parameters to be decoded.

    Returns
    -------
    list[QueriedComponent]
        Processed list of component dictionaries with decoded values.
        String values "none"/"null" are converted to None, "__all__" to slice(None).
        Tuple and list values are processed recursively with same conversions.
    """
    for queried_component in queried_components:
        for key, value in queried_component.items():
            if isinstance(value, str):
                if value.lower() in ["none", "null"]:
                    queried_component[key] = None
                elif value.lower() in ["__all__"]:
                    queried_component[key] = slice(None)

            elif isinstance(value, tuple):
                value = list(value)
                for i, value_i in enumerate(value):
                    if value_i in ["none", "null", "None", "Null"]:
                        value[i] = None
                    elif value_i in ["__all__"]:
                        value[i] = slice(None)
                queried_component[key] = tuple(value)

            elif isinstance(value, list):
                for i, value_i in enumerate(value):
                    if value_i in ["none", "null", "None", "Null"]:
                        value[i] = None
                    elif value_i in ["__all__"]:
                        value[i] = slice(None)
                    else:
                        value[i] = value_i
                queried_component[key] = tuple(value)

    return queried_components


def old_fnArgsDecoder(queried_components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Legacy function for decoding function arguments with expanded syntax support.

    Parameters
    ----------
    queried_components : list of dict
        List of component dictionaries containing query parameters to be decoded.

    Returns
    -------
    list of dict
        Processed list of component dictionaries with decoded values.
        Supports more complex string patterns including slice expressions.

    Notes
    -----
    This function is deprecated in favor of fnArgsDecoder. It provides backward
    compatibility for more complex slice notations and eval-based conversions.
    """
    for queried_component in queried_components:
        for key, value in queried_component.items():
            if isinstance(value, str):
                if value.lower() in [
                    "none",
                    "null",
                    "slice(none)",
                    "slice(null)",
                    "slice('none')",
                ]:
                    queried_component[key] = (
                        None if "slice" not in value else slice(None)
                    )
                elif "slice(None)" in value and value != "slice(None)":
                    queried_component[key] = eval(value)
                # if value in ["None", "null"]:
                #     queried_component[key] = None
                # elif value in ["slice(None)", "slice(null)", "slice('None')"]:
                #     queried_component[key] = slice(None)
                # elif value != 'slice(None)' and 'slice(None)' in value:
                #     # in case that llm should have returned a tuple ('slice(None)', 'slice(None)', "specific_index")
                #     # but returned a string "('slice(None)', 'slice(None)', "specific_index")"
                #     queried_component[key] = eval(value)

            elif isinstance(value, tuple):
                value = list(value)
                for i, value_i in enumerate(value):
                    if value_i in ["None", "null"]:
                        value[i] = None
                    elif value_i in ["slice(None)", "slice(null)", "slice('None')"]:
                        value[i] = slice(None)
                queried_component[key] = tuple(value)

            elif isinstance(value, list):
                for i, value_i in enumerate(value):
                    if value_i in ["None", "null"]:
                        value[i] = None
                    elif value_i in ["slice(None)", "slice(null)", "slice('None')"]:
                        value[i] = slice(None)
                    else:
                        value[i] = value_i
                queried_component[key] = tuple(value)

    return queried_components


def get_component_type(name: str, m: ModelDictWithPyomo) -> Optional[str]:
    """
    Determine the component type of a given component name in a model dictionary.

    Parameters
    ----------
    name : str
        The name of the component to look up.
    m : ModelDictWithPyomo
        Model dictionary containing component information organized by type.

    Returns
    -------
    str or None
        Component type ("parameters", "variables", "sets", "constraints", "objective")
        if found, None otherwise.
    """
    TYPES = ["parameters", "variables", "sets", "constraints", "objective"]
    return next((c_type for c_type in TYPES if name in m["components"][c_type]), None)


def get_new_model_name(queried_model: str) -> str:
    """
    Generate a new model name by incrementing the numeric suffix.

    Parameters
    ----------
    queried_model : str
        Current model name in the format "prefix_number".

    Returns
    -------
    str
        New model name with incremented numeric suffix.

    Examples
    --------
    >>> get_new_model_name("model_1")
    "model_2"
    """
    prefix, number = queried_model.rsplit("_", 1)
    incremented_number = int(number) + 1
    new_model_name = f"{prefix}_{incremented_number}"
    return new_model_name


def syntax_guidance(
    queried_function: str,
    queried_components: List[str],
    queried_model: str,
    models_dict: ModelsContainer,
) -> SyntaxGuidanceInternalResult:
    """
    Generate syntax guidance for function calls based on model components.

    Parameters
    ----------
    queried_function : str
        Name of the function to provide guidance for.
        Must be one of: "feasibility_restoration", "sensitivity_analysis",
        "components_retrieval", "evaluate_modification", "external_tools".
    queried_components : list of str
        List of component names to analyze for syntax guidance.
    queried_model : str
        Name of the model to query.
    models_dict : ModelsContainer
        Dictionary containing model information and component details.

    Returns
    -------
    SyntaxGuidanceInternalResult
        - Syntax output string with detailed guidance
        - Syntax mode indicating index complexity ("single", "multiple", "all", "none")

    Raises
    ------
    AssertionError
        If queried_function is not in the recognized function list.
    """

    FUNCTIONS = [
        "feasibility_restoration",
        "sensitivity_analysis",
        "components_retrieval",
        "evaluate_modification",
        "external_tools",
    ]
    assert (
        queried_function in FUNCTIONS
    ), f"Function {queried_function} is not recognized."
    if queried_function == "external_tools":
        return SyntaxGuidanceInternalResult(
            syntax_output="external_tools", syntax_mode="none"
        )

    model_dict = models_dict[queried_model]
    function_syntax = "function to call: " + queried_function + "\n\n"  #
    queried_model_syntax = "queried_model: " + queried_model + "\n\n"  #

    def get_index_guidance(pattern: Tuple | int) -> IndexGuidanceResult:
        """
        provide index guidance in terms of an indexed pattern,
        supplementary is 'evaluate_modification' or None
        """

        if isinstance(pattern, tuple):
            mode: str = "multiple"
            tuple_guidance: str = f"""
Return a tuple with dimensions to be {len(pattern)}.
You need to fill in the tuple with the specific indexes provided by the user,
and the rest of the indexes that are not specified should be "__all__" in string.

- Must return a tuple with {len(pattern)} elements
- "__all__" is placeholder that represents all indexes in a dimension that user didn't specify
- "__all__" can be inserted into the tuple multiple times if there are multiple dimensions that user didn't specify
Example: If the dimension of an indexed component is 3,
the first index is specified as 4, the second and third indexes are not specified by users,
then return the tuple (4, "__all__", "__all__")

- Must be careful with the order of indexes in the tuple by inspecting code
Example: If the dimension of an indexed component is 2,
the first dimension represents time, the second dimension represents location,
the user specifies the location to be "NY", the time is not specified by users,
then return the tuple ("__all__", "NY")

- Must be careful with the data type of each index in the tuple by inspecting code
Example: If the dimension of an indexed component is 2,
the first dimension represents time, the second dimension represents location,
the user specifies the time to be 9, the location to be "NY"
then return the tuple (9, "NY") instead of ('9', "NY")

**Note:** indexes are not specified unless the **exact** values are provided and the total number of values is less than the size of the dimension.
Descriptions like "for all the indexes", "from one to ten (but the size of dimension is 10)" are considered as "Not specified",
which MUST use "__all__" instead of enumerating all indexes.
"""
            return IndexGuidanceResult(guidance=tuple_guidance, mode=mode)

        elif isinstance(pattern, int) or isinstance(pattern, str):
            mode: str = "single"
            primitive_guidance: str = f"""
When specific index provided, fill in the type of {type(pattern)}
If no specific index provided, return "__all__" in string

- "__all__" is placeholder that represents all indexes that user didn't specify

**Note:** indexes are not specified unless the **exact** values are provided and the total number of values is less than the size of the dimension.
Descriptions like "for all the indexes", "from one to ten (but the size of dimension is 10)" are considered as "Not specified",
which MUST use "__all__" instead of enumerating all indexes.
"""
            return IndexGuidanceResult(guidance=primitive_guidance, mode=mode)
        else:
            raise TypeError("pattern must be a tuple or an int or a str")

    def get_complex_guidance(flag: bool) -> str:
        """
        Generate complex syntax guidance for handling multiple component indexes.

        Parameters
        ----------
        flag : bool
            Whether complex syntax guidance is needed. If True, returns detailed
            guidance for handling multiple indexes and prohibited changes.
            If False, returns empty string.

        Returns
        -------
        str
            Complex syntax guidance string with examples and instructions for:
            - Handling multiple indexes in the same dimension
            - Managing prohibited index changes
            - Proper queried_components structure formatting
            Returns empty string if flag is False.

        Notes
        -----
        This function provides guidance for complex scenarios where users specify:
        - Multiple specific indexes that should be handled separately
        - Constraints on which indexes can or cannot be modified
        - Examples using a demand parameter indexed by locations (a, b, c)

        Examples
        --------
        The guidance includes examples like:
        - "help me change demand of a and c" -> separate entries for each index
        - "change demand, but demand of b cannot be changed" -> exclude prohibited indexes
        """
        example = [
            {"component_name": "dem", "component_indexes": "a"},
            {"component_name": "dem", "component_indexes": "c"},
        ]
        cs = f"""
If the user provides multiple indexes that belong to the same dimension,
add every specified index separately in the queried_components.
Example: help me change demand of a and c.

If the user provides indexes that are prohibited from being changed,
add every permitted index separately in the queried_components.
Example: help me change demand, but please note that the demand of b cannot be changed.

dem is a 1-dim parameter that means demand, and dem is indexed by a, b, c, then,
queried_components: {example}"""
        if flag:
            return cs
        else:
            return ""

    def get_supplementary_guidance(fn: str) -> str:
        """
        Generate supplementary guidance for specific optimization functions.

        Parameters
        ----------
        fn : str
            The function name for which to provide supplementary guidance.
            Currently supports "evaluate_modification". Other function names
            return empty string.

        Returns
        -------
        str
            Supplementary guidance string with detailed instructions for the
            specified function. Returns empty string if function is not
            "evaluate_modification".

        Notes
        -----
        This function provides function-specific guidance that extends the basic
        syntax guidance. For "evaluate_modification", it provides detailed
        instructions on:

        - Default behavior when no modification extent is specified
        - Available mathematical operations (=, +, -, *, /)
        - Proper delta value specification
        - Examples of different modification patterns
        - Consistency requirements for parameter modifications

        The guidance includes practical examples covering:
        - Absolute changes ("change it to 5")
        - Relative increases/decreases ("increase by 5", "decrease by 5%")
        - Addition/subtraction operations ("have 5 more/less units")
        - Percentage-based modifications with proper multipliers

        Examples
        --------
        For "evaluate_modification", the guidance includes examples like:
        - "change it to 5" → operation: "=", delta: 5
        - "increase it by 5%" → operation: "*", delta: 1.05
        - "decrease it by 5%" → operation: "*", delta: 0.95
        - "have 5 more units" → operation: "+", delta: 5
        """
        if fn == "evaluate_modification":
            supplementary_guidance = """
When no specific modification extent provided, always return operation: "!" and delta: 0

Otherwise, choose one of the following operations: "+", "-", "*", "/", and fill in the delta value.
Demonstrations:
change it to 5: operation: "=", delta: 13;
increase it to 5: operation: "=", delta: 5;
increase it by 5: operation: "+", delta: 5;
increase it by 5%: operation: "*", delta: 1.05;
decrease it to 5: operation: "=", delta: 5;
decrease it by 5: operation: "-", delta: 5;
decrease it by 5%: operation: "*", delta: 0.95;
discount it by 5%: operation: "*", delta: 0.95;
have 5 more units: operation: "+", delta: 5;
have 5 less units: operation: "-", delta: 5;

Make sure the delta value is consistent with the positivity/negativity of the parameters being modified.
"""
            return supplementary_guidance
        else:
            return ""

    need_complex_syntax: bool = False
    ref = []
    syntax_mode = []
    for component_name in queried_components:
        component_type = get_component_type(component_name, model_dict)
        if model_dict["components"][component_type][component_name]["is_indexed"]:
            # component is indexed
            need_complex_syntax = True
            component_index_set = model_dict["components"][component_type][
                component_name
            ]["index_set"]
            component_pattern = random.choice(list(component_index_set))
            situation, mode_i = get_index_guidance(component_pattern)
        else:
            # component is not indexed
            situation = "always return null"
            mode_i = "none"
        ref.append({"component_name": component_name, "component_indexes": situation})
        syntax_mode.append(mode_i)

    queried_component_syntax = f"queried_components: {ref} \n\n"  #
    complex_syntax = get_complex_guidance(need_complex_syntax)  #
    supplementary = get_supplementary_guidance(queried_function)  #
    syntax_output = (
        function_syntax
        + queried_model_syntax
        + queried_component_syntax
        + complex_syntax
        + supplementary
    )

    syntax_mode = set(syntax_mode)
    if len(syntax_mode) > 1:
        syntax_mode = "all"
    else:
        syntax_mode = next(iter(syntax_mode))
    return SyntaxGuidanceInternalResult(
        syntax_output=syntax_output, syntax_mode=syntax_mode
    )


def feasibility_restoration(
    queried_components: list[QueriedComponent],
    queried_model: str,
    models_dict: ModelsContainer,
) -> str:
    """
    Restore feasibility of an infeasible optimization model by adjusting parameters.

    Parameters
    ----------
    queried_components : list[QueriedComponent]
        List of component dictionaries containing parameter names and indexes to modify.
        Each dict should have 'component_name' and 'component_indexes' keys.
    queried_model : str
        Name of the infeasible model to restore.
    models_dict : ModelsContainer
        Dictionary containing all model information and instances.

    Returns
    -------
    str
        Feedback message describing the feasibility restoration results, including:
        - Parameter changes needed to restore feasibility
        - New model name and status
        - Analysis recommendations for user

    Notes
    -----
    - Only works on models with infeasible or infeasibleOrUnbounded status
    - Adds positive and negative slack variables to RHS parameters
    - Solves slack minimization problem to find minimal feasible changes
    - Creates new model instance with "_n+1" suffix
    - Has 5-minute time limit for solving
    """
    queried_model_dict = models_dict[queried_model]

    if queried_model_dict["model_status"] not in [
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ]:
        return "The model is not infeasible. No need to restore feasibility. Please confirm with the user."

    # define slack parameters
    model: pe.ConcreteModel = queried_model_dict["model_class"].clone()
    for component in queried_components:
        param_name: str = component["component_name"]
        param_indexes = component["component_indexes"]
        logger.debug(f"param_name: {param_name}, param_indexes: {param_indexes}")

        component_type = get_component_type(param_name, queried_model_dict)
        if component_type == "parameters":
            if isinstance(param_indexes, tuple):
                eval_param = eval(f"model.{param_name}")
                if len(eval_param[param_indexes].index()) <= 0:
                    raise IndexError(
                        (
                            "Error: Indexes are not valid. This usually happens "
                            "when the order of indexes in the tuple is incorrect."
                        )
                    )

            if queried_model_dict["components"]["parameters"][param_name]["is_RHS"]:
                # First, add slacks to all indexes and fix all of them as 0
                exec(
                    "model.slack_pos_"
                    + param_name
                    + "=pe.Var(model."
                    + param_name
                    + ".index_set(), within=pe.NonNegativeReals)"
                )
                exec(
                    "model.slack_neg_"
                    + param_name
                    + "=pe.Var(model."
                    + param_name
                    + ".index_set(), within=pe.NonNegativeReals)"
                )
                model_slack_pos_param = eval("model.slack_pos_" + param_name)
                model_slack_neg_param = eval("model.slack_neg_" + param_name)
                model_slack_pos_param.fix(0)
                model_slack_neg_param.fix(0)
                # Second, unfix the slacks for the specific indexes provided in the query
                model_slack_pos_param[param_indexes].unfix()
                model_slack_neg_param[param_indexes].unfix()

            else:
                feedback = f"""
Feedback from internal tools:
Warning. {param_name} is not a RHS parameter in the model.
This parameter is LHS parameter.
Changing LHS parameter for feasibility restoration without specifying modification extent
can extend solving time and risk terminating the optimization process prematurely before finding an optimal solution.
Users need to try other parameters for feasibility restoration,
or specify a modification extent (e.g., a 5% increase) to directly assess the impact of this modification, if they are particularly interested in this parameter.
"""
                return feedback
        else:
            wrong_component_type = component_type
            feedback = f"""
Feedback from internal tools:
Error. {param_name} is not a parameter in the model but a {wrong_component_type}.
Users need to provide a valid parameter for feasibility restoration."""
            return feedback

    # generate replacements
    iis_param = []
    replacements_list = []
    for component in queried_components:
        param_name = component["component_name"]
        param_indexes = component["component_indexes"]
        for idx in eval("model." + param_name + ".index_set()"):
            model_param = eval("model." + param_name)
            iis_param.append((param_name, idx))
            expr_param = model_param[idx]
            slack_var_pos = eval("model.slack_pos_" + param_name)[idx]
            slack_var_neg = eval("model.slack_neg_" + param_name)[idx]
            replacements = {id(expr_param): expr_param + slack_var_pos - slack_var_neg}
            replacements_list.append(replacements)

    # replace constraints
    original_consts: list[IndexedConstraint] = []
    for consts_name, consts in model.component_map(pe.Constraint).items():
        original_consts.append(consts)

    model.slack_iis_constraints = pe.ConstraintList()
    for consts in original_consts:
        for const_idx in consts.index_set():
            try:
                const: ConstraintData = consts[const_idx]
                new_expr = clone_expression(const.expr)
                for replacements in replacements_list:
                    new_expr = replace_expressions(new_expr, replacements)
                model.slack_iis_constraints.add(new_expr)
                const.deactivate()
            except Exception as e:
                logger.error(f"Skip the skipped constraint: {e}")

    # replace objective
    objectives = model.component_objects(pe.Objective, active=True)
    for obj in objectives:
        obj.deactivate()

    # minimize the 1-norm of the slacks that are added
    new_obj = 0
    for p, idx in iis_param:
        slack_var_pos = eval("model.slack_pos_" + p)[idx]
        slack_var_neg = eval("model.slack_neg_" + p)[idx]
        new_obj += slack_var_pos + slack_var_neg

    model.slack_obj = pe.Objective(expr=new_obj, sense=pe.minimize)
    # solve the model
    opt = SolverFactory("gurobi")
    opt.options["nonConvex"] = 2
    opt.options["TimeLimit"] = 300  # 5min time limit
    results = opt.solve(model, tee=True)
    # construct technical feedback
    termination_condition = results.solver.termination_condition
    # new_model_dict = pyomo2json(model, termination_condition=termination_condition)
    new_model_dict = copy.deepcopy(queried_model_dict)
    feedback = f"The following changes are made to {queried_model}: \n"
    description = f"a model with the following changes to {queried_model}: \n"
    new_model_name = get_new_model_name(queried_model)

    if termination_condition == TerminationCondition.maxTimeLimit:
        for p, idx in iis_param:
            feedback = feedback + f"attempt to change {p} at {idx}; \n"
            description = description + f"attempt to change {p}{idx}; \n"
        feedback = feedback + "\n\nThe model cannot be solved due to time limit."
        description = (
            description[:7]
            + ", which cannot be solved due to time limit."
            + description[7:]
        )
        new_model_dict["model_description"] = description
        new_model_dict["model_status"] = TerminationCondition.maxTimeLimit
        models_dict[new_model_name] = new_model_dict
    elif termination_condition == TerminationCondition.optimal:
        for p, idx in iis_param:
            slack_var_pos = eval("model.slack_pos_" + p)[idx].value
            slack_var_neg = eval("model.slack_neg_" + p)[idx].value
            idx = "" if idx is None else f" at {idx}"
            if slack_var_pos > 1e-5:
                feedback = feedback + f"change {p}{idx} by +{slack_var_pos} unit; \n"
                description = (
                    description + f"change {p}{idx} by +{slack_var_pos} unit; \n"
                )
            elif slack_var_neg > 1e-5:
                feedback = feedback + f"change {p}{idx} by -{slack_var_neg} unit; \n"
                description = (
                    description + f"change {p}{idx} by -{slack_var_neg} unit; \n"
                )
        feedback = feedback + "\n\nThe model now becomes feasible. "
        feedback = (
            feedback
            + "\n\nHelp the user analyze why the feasibility can be restored by these changes. Let user know this new model will be referred to as {new_model_name}."
        )
        description = description[:7] + ", which becomes feasible" + description[7:]
        new_model_dict["model_description"] = description
        new_model_dict["model_status"] = TerminationCondition.optimal
        models_dict[new_model_name] = new_model_dict
    else:
        feedback = "The model remains infeasible after only changing the following: \n"
        for p, idx in iis_param:
            idx = "" if idx is None else f" at {idx}"
            feedback = feedback + f"{p}{idx}; \n"
        description = feedback
        models_dict[queried_model]["model_description"] = description
        feedback = (
            feedback
            + "\n\nThis is determined by the nature of the model, rather than an error of internal tools. Help the user analyze why the feasibility is not restored."
        )

    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def sensitivity_analysis(
    queried_components: list[QueriedComponent],
    queried_model: str,
    models_dict: ModelsContainer,
) -> str:
    """
    Perform sensitivity analysis on linear programming model parameters.

    Parameters
    ----------
    queried_components : list of dict
        List of component dictionaries containing parameter names and indexes to analyze.
        Each dict should have 'component_name' and 'component_indexes' keys.
    queried_model : str
        Name of the model to analyze.
    models_dict : ModelsContainer
        Dictionary containing all model information and instances.

    Returns
    -------
    str
        Feedback message containing sensitivity analysis results:
        - Impact of parameter perturbations on optimal objective value
        - Dual value calculations and interpretations
        - Recommendations for user analysis

    Notes
    -----
    - Only works on feasible linear programming models
    - Only supports RHS parameters (right-hand side of constraints)
    - Computes dual values to determine sensitivity
    - Uses symbolic differentiation to find parameter coefficients
    - Automatically solves model with dual suffixes if not already present
    """
    queried_model_dict = models_dict[queried_model]
    model: pe.ConcreteModel = queried_model_dict["model_class"].clone()

    if queried_model_dict["model_status"] in [
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ]:
        feedback = "Error: The model is infeasible. Sensitivity analysis cannot be performed on an infeasible model."
        feedback = "Feedback from internal tools: \n" + feedback
        return feedback
    if queried_model_dict["model_type"] != "LP":
        feedback = "Error: The model is not a linear programming model. Internal tools do not support sensitivity analysis on other types of models."
        feedback = "Feedback from internal tools: \n" + feedback
        return feedback

    def locate_param(
        param_name: str, idx: Any, model: Any = model
    ) -> List[Dict[str, Any]]:
        """
        Locate constraints containing a specific parameter and compute its coefficients.

        This function identifies all constraints in which a given parameter (at a specific
        index) appears and calculates the coefficient of that parameter in each constraint's
        expression using symbolic differentiation.

        Parameters
        ----------
        param_name : str
            Name of the parameter to locate within constraints.
        idx : Any
            Index of the parameter instance to analyze. Can be int, str, tuple,
            or None for non-indexed parameters.
        model : Any, optional
            Pyomo model object containing the constraints and parameters.
            Defaults to the model from outer scope.

        Returns
        -------
        list of dict
            List of dictionaries, each containing information about a constraint
            that includes the specified parameter. Each dictionary has keys:
            - 'const_name' : str
                Name of the constraint containing the parameter
            - 'const_indexes' : Any
                Index of the specific constraint instance
            - 'coefficient' : float
                Coefficient of the parameter in the constraint expression,
                computed through symbolic differentiation

        Notes
        -----
        - Uses symbolic differentiation to compute parameter coefficients
        - Handles constraint expressions with body, lower, and upper bounds
        - Coefficient is calculated as: -(∂body/∂param) + (∂lower/∂param) + (∂upper/∂param)
        - The negative sign on body derivative accounts for standard constraint form
        - Only processes constraints listed in the parameter's 'cons_in' metadata
        - Breaks after finding the parameter in each constraint to avoid duplicates

        Examples
        --------
        For a parameter 'demand' at index 'location_A' that appears in constraints
        'supply_balance' and 'capacity_limit', this function returns:

        [
            {
                'const_name': 'supply_balance',
                'const_indexes': ('location_A', 'time_1'),
                'coefficient': 1.0
            },
            {
                'const_name': 'capacity_limit',
                'const_indexes': ('location_A',),
                'coefficient': -0.5
            }
        ]
        """
        in_consts: List[Dict[str, Any]] = []
        param_name_idx = str(eval("model." + param_name)[idx])
        for const_name in queried_model_dict["components"]["parameters"][param_name][
            "cons_in"
        ]:
            model_const: IndexedConstraint = eval("model." + const_name)
            for con_idx in model_const.index_set():
                con_i: ConstraintData = model_const[con_idx]
                expr_params = identify_mutable_parameters(con_i.expr)
                for expr_param in expr_params:
                    if expr_param.name == param_name_idx:
                        coef_body = -differentiate(
                            con_i.body, wrt=expr_param, mode=Modes.reverse_symbolic
                        )  # type: ignore
                        coef_lower = differentiate(
                            con_i.lower, wrt=expr_param, mode=Modes.reverse_symbolic
                        )
                        coef_upper = differentiate(
                            con_i.upper, wrt=expr_param, mode=Modes.reverse_symbolic
                        )
                        coef = coef_body + coef_lower + coef_upper
                        in_consts.append(
                            {
                                "const_name": const_name,
                                "const_indexes": con_idx,
                                "coefficient": coef,
                            }
                        )
                        break

        return in_consts

    param_const_pairs = []
    for component in queried_components:
        param_name = component["component_name"]
        param_indexes = component["component_indexes"]
        logger.debug(f"param_name: {param_name}, param_indexes: {param_indexes}")
        component_type = get_component_type(param_name, queried_model_dict)
        if component_type == "parameters":

            if isinstance(param_indexes, tuple):
                eval_param = eval(f"model.{param_name}")
                if len(eval_param[param_indexes].index()) <= 0:
                    raise IndexError(
                        (
                            "Error: Indexes are not valid. This usually happens "
                            "when the order of indexes in the tuple is incorrect."
                        )
                    )

            if queried_model_dict["components"]["parameters"][param_name]["is_RHS"]:
                if isinstance(param_indexes, tuple):
                    if slice(None) in param_indexes:
                        for model_param_i in eval("model." + param_name)[param_indexes]:
                            model_param_i_indexes = model_param_i.index()
                            param_const_pair = {
                                "param_name": param_name,
                                "param_indexes": model_param_i_indexes,
                                "consts": locate_param(
                                    param_name, model_param_i_indexes
                                ),
                            }
                            param_const_pairs.append(param_const_pair)
                    else:
                        param_const_pair = {
                            "param_name": param_name,
                            "param_indexes": param_indexes,
                            "consts": locate_param(param_name, param_indexes),
                        }
                        param_const_pairs.append(param_const_pair)
                elif isinstance(param_indexes, slice):
                    for model_param_i in eval("model." + param_name)[param_indexes]:
                        model_param_i_indexes = model_param_i.index()
                        param_const_pair = {
                            "param_name": param_name,
                            "param_indexes": model_param_i_indexes,
                            "consts": locate_param(param_name, model_param_i_indexes),
                        }
                        param_const_pairs.append(param_const_pair)
                elif isinstance(param_indexes, int) or isinstance(param_indexes, str):
                    param_const_pair = {
                        "param_name": param_name,
                        "param_indexes": param_indexes,
                        "consts": locate_param(param_name, param_indexes),
                    }
                    param_const_pairs.append(param_const_pair)
                elif param_indexes is None:
                    param_const_pair = {
                        "param_name": param_name,
                        "param_indexes": param_indexes,
                        "consts": locate_param(param_name, param_indexes),
                    }
                    param_const_pairs.append(param_const_pair)

            else:
                feedback = f"Error: {param_name} is not a RHS parameter in the model. "
                feedback += """
Please confirm with the user and ask them to provide a valid RHS parameter for sensitivity analysis,
or if they are particularly interested in these parameters, they must specify a modification extent (e.g., a 5% increase) to directly assess the impact of this modification."""
                feedback = "Feedback from internal tools: \n" + feedback
                return feedback

        else:
            wrong_component_type = component_type
            feedback = f"Error: {param_name} is not a parameter in the model but a {wrong_component_type}. "
            feedback += """Please confirm with the user and ask them to provide a valid parameter for sensitivity analysis."""
            feedback = "Feedback from internal tools: \n" + feedback
            return feedback

    # duals = []
    logger.debug(
        f' Does this model have model.dual? {model.find_component("dual") is None}'
    )
    if model.find_component("dual") is None:
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
        opt = SolverFactory("gurobi")
        results = opt.solve(model, tee=True)
        termination_condition = results.solver.termination_condition
        # update the models_dict
        models_dict[queried_model]["model_class"] = model

    for param_const_pair in param_const_pairs:
        for const in param_const_pair["consts"]:
            const_name = const["const_name"]
            const_indexes = const["const_indexes"]
            const_coef = const["coefficient"]
            model_const = eval("model." + const_name)
            model_const_i = model_const[const_indexes]
            const["dual_value"] = const_coef * model.dual[model_const_i]
            # print(f'const_name: {const_name}, const_indexes: {const_indexes}, const_coef: {const_coef}, dual_value: {const["dual_value"]}')
            # break

    # construct feedback
    feedback = "The sensitivity analysis results are as follows: \n"
    for param_const_pair in param_const_pairs:
        param_name = param_const_pair["param_name"]
        param_indexes = param_const_pair["param_indexes"]
        param_indexes = f" at {param_indexes}" if param_indexes is not None else ""
        feedback = (
            feedback
            + f"when a small positive perturbation is made to {param_name}{param_indexes}, "
        )
        total_value = 0
        for const in param_const_pair["consts"]:
            # print(f'retrieving dual value of {const["const_name"]} at {const["const_indexes"]}')
            dual_value = const["dual_value"]
            # print(f'the dual value of {const["const_name"]} at {const["const_indexes"]} is {dual_value}')
            total_value += dual_value

        if total_value > 1e-5:
            feedback = (
                feedback
                + f"the optimal objective value will change by {total_value} unit\n"
            )
        elif total_value < -1e-5:
            feedback = (
                feedback
                + f"the optimal objective value will change by {total_value} unit\n"
            )
        else:
            feedback = feedback + "the optimal objective value will not change \n"

    feedback += "Please explain these results to the user. \n"
    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def components_retrieval(
    queried_components: list[QueriedComponent],
    queried_model: str,
    models_dict: ModelsContainer,
) -> str:
    """
    Retrieve current values or expressions of model components.

    Parameters
    ----------
    queried_components : list[QueriedComponent]
        List of component dictionaries containing component names and indexes to
        retrieve. Each dict should have 'component_name' and 'component_indexes' keys.
    queried_model : str
        Name of the model to query.
    models_dict : ModelsContainer
        Dictionary containing all model information and instances.

    Returns
    -------
    str
        Feedback message containing component values and expressions:
        - Parameter values
        - Variable values
        - Set data
        - Constraint expressions
        - Objective function values
        - Recommendations for interpretation

    Notes
    -----
    - Supports all component types: parameters, variables, sets, constraints, objectives
    - Handles indexed and non-indexed components
    - Works with tuple, slice, int, str, and None index specifications
    - Provides human-readable descriptions with physical meanings
    """
    queried_model_dict = models_dict[queried_model]
    model: pe.ConcreteModel = queried_model_dict["model_class"].clone()  # noqa: F841
    feedback = f"In the {queried_model}, "
    for component in queried_components:
        component_name = component["component_name"]
        component_indexes = component["component_indexes"]
        logger.debug(
            f"component_name: {component_name}, component_indexes: {component_indexes}"
        )
        model_component = eval("model." + component_name)

        if isinstance(component_indexes, tuple):
            # supposed to retrieve multiple components
            if slice(None) in component_indexes:

                if len(model_component[component_indexes].index()) <= 0:
                    raise IndexError(
                        (
                            "Error: Indexes are not valid. This usually happens "
                            "when the order of indexes in the tuple is incorrect."
                        )
                    )

                for model_component_i in model_component[component_indexes]:
                    model_component_i_indexes = model_component_i.index()
                    feedback = (
                        feedback
                        + f"{component_name} at {str(model_component_i_indexes)} is "
                    )
                    component_retrieval = ""
                    if component_name in queried_model_dict["components"]["parameters"]:
                        component_retrieval = str(model_component_i.value)
                    elif (
                        component_name in queried_model_dict["components"]["variables"]
                    ):
                        component_retrieval = str(model_component_i.value)
                    elif component_name in queried_model_dict["components"]["sets"]:
                        component_retrieval = str(model_component_i.data())
                    elif (
                        component_name
                        in queried_model_dict["components"]["constraints"]
                    ):
                        component_retrieval = str(model_component_i.expr)
                    elif (
                        component_name in queried_model_dict["components"]["objective"]
                    ):
                        component_retrieval = str(model_component_i())
                    feedback = feedback + f"{component_retrieval}.\n"

            else:
                # supposed to retrieve one component
                feedback = (
                    feedback + f"{component_name} at {str(component_indexes)} is "
                )
                component_retrieval = ""
                if component_name in queried_model_dict["components"]["parameters"]:
                    component_retrieval = str(model_component[component_indexes].value)
                elif component_name in queried_model_dict["components"]["variables"]:
                    component_retrieval = str(model_component[component_indexes].value)
                elif component_name in queried_model_dict["components"]["sets"]:
                    component_retrieval = str(model_component[component_indexes].data())
                elif component_name in queried_model_dict["components"]["constraints"]:
                    component_retrieval = str(model_component[component_indexes].expr)
                elif component_name in queried_model_dict["components"]["objective"]:
                    component_retrieval = str(model_component[component_indexes]())
                feedback = feedback + f"{component_retrieval}.\n"

        elif isinstance(component_indexes, slice):
            # supposed to retrieve multiple components
            for model_component_i in model_component[component_indexes]:
                model_component_i_indexes = model_component_i.index()
                feedback = (
                    feedback
                    + f"{component_name} at {str(model_component_i_indexes)} is "
                )
                component_retrieval = ""
                if component_name in queried_model_dict["components"]["parameters"]:
                    component_retrieval = str(model_component_i.value)
                elif component_name in queried_model_dict["components"]["variables"]:
                    component_retrieval = str(model_component_i.value)
                elif component_name in queried_model_dict["components"]["sets"]:
                    component_retrieval = str(model_component_i.data())
                elif component_name in queried_model_dict["components"]["constraints"]:
                    component_retrieval = str(model_component_i.expr)
                elif component_name in queried_model_dict["components"]["objective"]:
                    component_retrieval = str(model_component_i())
                feedback = feedback + f"{component_retrieval}.\n"

        elif isinstance(component_indexes, int) or isinstance(component_indexes, str):
            # supposed to retrieve one component
            feedback = feedback + f"{component_name} at {str(component_indexes)} is "
            component_retrieval = ""
            if component_name in queried_model_dict["components"]["parameters"]:
                component_retrieval = str(model_component[component_indexes].value)
            elif component_name in queried_model_dict["components"]["variables"]:
                component_retrieval = str(model_component[component_indexes].value)
            elif component_name in queried_model_dict["components"]["sets"]:
                component_retrieval = str(model_component[component_indexes].data())
            elif component_name in queried_model_dict["components"]["constraints"]:
                component_retrieval = str(model_component[component_indexes].expr)
            elif component_name in queried_model_dict["components"]["objective"]:
                component_retrieval = str(model_component[component_indexes]())
            feedback = feedback + f"{component_retrieval}.\n"

        elif component_indexes is None:
            # supposed to retrieve one component
            feedback = feedback + f"{component_name} is "
            component_retrieval = ""
            if component_name in queried_model_dict["components"]["parameters"]:
                component_retrieval = str(model_component.value)
            elif component_name in queried_model_dict["components"]["variables"]:
                component_retrieval = str(model_component.value)
            elif component_name in queried_model_dict["components"]["sets"]:
                component_retrieval = str(model_component.data())
            elif component_name in queried_model_dict["components"]["constraints"]:
                component_retrieval = str(model_component.expr)
            elif component_name in queried_model_dict["components"]["objective"]:
                component_retrieval = str(model_component())
            feedback = feedback + f"{component_retrieval}.\n"

    feedback += (
        "Please describe the information using their physical meanings to the user. \n"
    )
    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def evaluate_modification(
    queried_components: list[QueriedComponent],
    queried_model: str,
    models_dict: ModelsContainer,
) -> str:
    """
    Evaluate the impact of specific parameter modifications on model behavior.

    Parameters
    ----------
    queried_components : list[QueriedComponent]
        List of component dictionaries containing modification specifications.
        Each dict should have keys: 'component_name', 'component_indexes',
        'operation', and 'delta'.
    queried_model : str
        Name of the model to modify and evaluate.
    models_dict : ModelsContainer
        Dictionary containing all model information and instances.

    Returns
    -------
    str
        Feedback message containing evaluation results:
        - Description of modifications made
        - New model status and objective value
        - Comparison with original model performance
        - Analysis recommendations for user

    Notes
    -----
    - Requires specific modification extents (operation and delta values)
    - Supports operations: "=", "+", "-", "*", "/" with numeric delta
    - Parameters are changed, variables are fixed to new values
    - Creates new model instance with "_n+1" suffix
    - Solves modified model with 5-minute time limit
    - Handles feasible, infeasible, and time-limited results
    """
    queried_model_dict = models_dict[queried_model]
    model: pe.ConcreteModel = queried_model_dict["model_class"].clone()
    original_obj_value = None
    for obj_name, obj in model.component_map(pe.Objective).items():
        original_obj_value = queried_model_dict["components"]["objective"][obj_name][
            "optimal_value"
        ]

    feedback = f"In the {queried_model}, the following modifications are made: \n"
    description = f"a model with the following changes to {queried_model}: \n"
    for component in queried_components:
        # validate the component dictionary
        assert "operation" in component
        assert "delta" in component

        component_name = component["component_name"]
        component_indexes = component["component_indexes"]
        component_operation = component["operation"]

        if component_operation == "!":
            return (
                "Error: The evaluate_modification function requires a specific "
                "modification extent. Debug suggestion: distribute this task to "
                "operator again and ask them to use sensitivity_analysis "
                "function instead."
            )

        component_delta = str(component["delta"])
        logger.debug(
            f"component_name: {component_name}, "
            f"component_indexes: {component_indexes}, "
            f"component_operation: {component_operation}, "
            f"component_delta: {component_delta}"
        )
        model_component = eval("model." + component_name)

        if isinstance(component_indexes, tuple):
            if slice(None) in component_indexes:

                if len(model_component[component_indexes].index()) <= 0:
                    raise IndexError(
                        (
                            "Error: Indexes are not valid. This usually happens "
                            "when the order of indexes in the tuple is incorrect."
                        )
                    )

                for model_component_i in model_component[component_indexes]:
                    model_component_i_indexes = model_component_i.index()
                    if component_name in queried_model_dict["components"]["parameters"]:
                        value_for_modification = eval("model." + component_name)[
                            model_component_i_indexes
                        ].value
                        value_after_modification = (
                            eval(component_delta)
                            if component_operation == "="
                            else eval(
                                str(value_for_modification)
                                + component_operation
                                + component_delta
                            )
                        )
                        model_component[model_component_i_indexes].set_value(
                            value_after_modification
                        )
                        changed_or_fixed = " is changed to "
                    elif (
                        component_name in queried_model_dict["components"]["variables"]
                    ):
                        value_for_modification = eval("model." + component_name)[
                            model_component_i_indexes
                        ].value
                        value_after_modification = (
                            eval(component_delta)
                            if component_operation == "="
                            else eval(
                                str(value_for_modification)
                                + component_operation
                                + component_delta
                            )
                        )
                        model_component[model_component_i_indexes].fix(
                            value_after_modification
                        )
                        changed_or_fixed = " is fixed to "
                    else:
                        continue

                    feedback += (
                        f"{component_name} at {str(model_component_i_indexes)}"
                        + changed_or_fixed
                        + str(value_after_modification)
                        + ".\n"
                    )
                    description += (
                        f"{component_name} at {str(model_component_i_indexes)}"
                        + changed_or_fixed
                        + str(value_after_modification)
                        + ".\n"
                    )
            else:
                if component_name in queried_model_dict["components"]["parameters"]:
                    value_for_modification = eval("model." + component_name)[
                        component_indexes
                    ].value
                    value_after_modification = (
                        eval(component_delta)
                        if component_operation == "="
                        else eval(
                            str(value_for_modification)
                            + component_operation
                            + component_delta
                        )
                    )
                    model_component[component_indexes].set_value(
                        value_after_modification
                    )
                    changed_or_fixed = " is changed to "
                elif component_name in queried_model_dict["components"]["variables"]:
                    value_for_modification = eval("model." + component_name)[
                        component_indexes
                    ].value
                    value_after_modification = (
                        eval(component_delta)
                        if component_operation == "="
                        else eval(
                            str(value_for_modification)
                            + component_operation
                            + component_delta
                        )
                    )
                    model_component[component_indexes].fix(value_after_modification)
                    changed_or_fixed = " is fixed to "
                else:
                    raise ValueError("Component not found or not modifiable.")

                feedback += (
                    f"{component_name} at {str(component_indexes)}"
                    + changed_or_fixed
                    + str(value_after_modification)
                    + ".\n"
                )
                description += (
                    f"{component_name} at {str(component_indexes)}"
                    + changed_or_fixed
                    + str(value_after_modification)
                    + ".\n"
                )

        elif isinstance(component_indexes, slice):
            for model_component_i in model_component[component_indexes]:
                model_component_i_indexes = model_component_i.index()
                if component_name in queried_model_dict["components"]["parameters"]:
                    value_for_modification = eval("model." + component_name)[
                        model_component_i_indexes
                    ].value
                    value_after_modification = (
                        eval(component_delta)
                        if component_operation == "="
                        else eval(
                            str(value_for_modification)
                            + component_operation
                            + component_delta
                        )
                    )
                    model_component[model_component_i_indexes].set_value(
                        value_after_modification
                    )
                    changed_or_fixed = " is changed to "
                elif component_name in queried_model_dict["components"]["variables"]:
                    value_for_modification = eval("model." + component_name)[
                        model_component_i_indexes
                    ].value
                    value_after_modification = (
                        eval(component_delta)
                        if component_operation == "="
                        else eval(
                            str(value_for_modification)
                            + component_operation
                            + component_delta
                        )
                    )
                    model_component[model_component_i_indexes].fix(
                        value_after_modification
                    )
                    changed_or_fixed = " is fixed to "
                else:
                    continue

                feedback += (
                    f"{component_name} at {str(model_component_i_indexes)}"
                    + changed_or_fixed
                    + str(value_after_modification)
                    + ".\n"
                )
                description += (
                    f"{component_name} at {str(model_component_i_indexes)}"
                    + changed_or_fixed
                    + str(value_after_modification)
                    + ".\n"
                )

        elif (
            isinstance(component_indexes, int)
            or isinstance(component_indexes, str)
            or component_indexes is None
        ):
            if component_name in queried_model_dict["components"]["parameters"]:
                value_for_modification = eval("model." + component_name)[
                    component_indexes
                ].value
                value_after_modification = (
                    eval(component_delta)
                    if component_operation == "="
                    else eval(
                        str(value_for_modification)
                        + component_operation
                        + component_delta
                    )
                )
                model_component[component_indexes].set_value(value_after_modification)
                changed_or_fixed = " is changed to "
            elif component_name in queried_model_dict["components"]["variables"]:
                value_for_modification = eval("model." + component_name)[
                    component_indexes
                ].value
                value_after_modification = (
                    eval(component_delta)
                    if component_operation == "="
                    else eval(
                        str(value_for_modification)
                        + component_operation
                        + component_delta
                    )
                )
                model_component[component_indexes].fix(value_after_modification)
                changed_or_fixed = " is fixed to "
            else:
                raise ValueError("Component not found or not modifiable.")

            idx = "" if component_indexes is None else f" at {str(component_indexes)}"
            feedback += (
                f"{component_name}{idx}"
                + changed_or_fixed
                + str(value_after_modification)
                + ".\n"
            )
            description += (
                f"{component_name}{idx}"
                + changed_or_fixed
                + str(value_after_modification)
                + ".\n"
            )

        # resolve the model
        opt = SolverFactory("gurobi")
        opt.options["nonConvex"] = 2
        opt.options["TimeLimit"] = 300  # 5min time limit
        results = opt.solve(model, tee=True)
        # construct technical feedback
        termination_condition = results.solver.termination_condition
        new_model_dict = pyomo2json(model, termination_condition=termination_condition)

        new_model_name = get_new_model_name(queried_model)

        if termination_condition == TerminationCondition.maxTimeLimit:
            feedback += f"\n\nThe model is not solved due to time limit, and the new model will be referred to as {new_model_name}."
            feedback += f"The best objective value found so far is {results.Problem[0]['Upper bound']}.\n"
            feedback = (
                feedback
                + "\n\nHelp the user analyze the influence of these modifications."
            )
            description = (
                description[:7]
                + ", which is not solved due to time limit"
                + description[7:]
            )
            new_model_dict["model_description"] = description
            models_dict[new_model_name] = new_model_dict
        elif termination_condition == TerminationCondition.optimal:
            feedback += f"\n\nThe model now is feasible, and the new model will be referred to as {new_model_name}."
            feedback += f"The optimal objective value found is {results.Problem[0]['Lower bound']}.\n"
            feedback = (
                feedback
                + "\n\nHelp the user analyze the influence of these modifications. "
            )
            description = description[:7] + ", which is feasible" + description[7:]
            new_model_dict["model_description"] = description
            models_dict[new_model_name] = new_model_dict
        else:
            feedback += f"\n\nThe model now is infeasible, and the new model will be referred to as {new_model_name}."
            feedback = (
                feedback
                + "\n\nHelp the user analyze the influence of these modifications."
            )
            description = description[:7] + ", which is infeasible" + description[7:]
            new_model_dict["model_description"] = description
            models_dict[new_model_name] = new_model_dict

        reminder = f"Reminder: the status of old model, {queried_model}, was "
        if queried_model_dict["model_status"] == TerminationCondition.maxTimeLimit:
            reminder += f"is not solved due to time limit with best objective value found as {original_obj_value}."
        elif queried_model_dict["model_status"] == TerminationCondition.optimal:
            reminder += (
                f"solved with optimal objective value found as {original_obj_value}."
            )
        elif queried_model_dict["model_status"] in [
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            reminder += "infeasible."

        feedback += reminder
        feedback = "Feedback from internal tools: \n" + feedback

    return feedback
