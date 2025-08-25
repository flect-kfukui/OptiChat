import copy
import importlib
import importlib.util
import io
import os
import re
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, Generator, List, Set, Tuple

import pyomo.environ as pe
from loguru import logger
from pyomo.contrib import iis
from pyomo.core.base.constraint import ConstraintData, IndexedConstraint
from pyomo.core.base.objective import ScalarObjective
from pyomo.core.base.var import IndexedVar, VarData
from pyomo.core.expr.visitor import identify_mutable_parameters, identify_variables
from pyomo.opt import SolverFactory, TerminationCondition
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import TypedDict definitions from optichat_types
from optichat_types import (
    ComponentsContainerSerializable,
    ModelDictSerializable,
    ModelDictWithPyomo,
    ModelsContainer,
)


def validate_model_dict_structure(model_dict: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary matches the expected ModelDictWithPyomo structure.

    This function checks for the presence of required keys that are created
    by pyomo2json and modified by subsequent processing functions.

    Parameters
    ----------
    model_dict : dict
        Dictionary to validate against ModelDictWithPyomo structure

    Returns
    -------
    bool
        True if structure matches expected format, False otherwise
    """
    required_keys = {
        "model_class",
        "model_status",
        "model_type",
        "model_description",
        "components",
    }

    # Check required keys
    if not all(key in model_dict for key in required_keys):
        return False

    # Check components structure
    if "components" in model_dict:
        components = model_dict["components"]
        if not isinstance(components, dict):
            return False

        expected_component_types = {
            "sets",
            "parameters",
            "variables",
            "constraints",
            "objective",
        }
        if not all(comp_type in components for comp_type in expected_component_types):
            return False

    return True


def validate_serializable_model_dict_structure(model_dict: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary matches the expected ModelDictSerializable structure.

    This function checks for serializable model dictionaries that don't contain
    Pyomo objects (no model_class field, no index_set in components).

    Parameters
    ----------
    model_dict : dict
        Dictionary to validate against ModelDictSerializable structure

    Returns
    -------
    bool
        True if structure matches expected format, False otherwise
    """
    required_keys = {
        "model_status",
        "model_type",
        "model_description",
        "components",
    }

    # Check required keys (note: no model_class for serializable version)
    if not all(key in model_dict for key in required_keys):
        return False

    # Check components structure
    if "components" in model_dict:
        components = model_dict["components"]
        if not isinstance(components, dict):
            return False

        expected_component_types = {
            "sets",
            "parameters",
            "variables",
            "constraints",
            "objective",
        }
        if not all(comp_type in components for comp_type in expected_component_types):
            return False

    return True


def find_lhs_params(
    constraint_expr: str, param_names: List[str], var_names: List[str]
) -> Set[str]:
    """
    Find parameters that appear on the left-hand side (LHS) of constraints.

    Parameters
    ----------
    constraint_expr : str
        String representation of the constraint expression to analyze.
    param_names : list of str
        List of parameter names to search for in the constraint.
    var_names : list of str
        List of variable names to search for in the constraint.

    Returns
    -------
    set
        Set of parameter names that appear on the LHS of the constraint,
        identified by their multiplicative relationship with variables.

    Notes
    -----
    Analyzes constraint expressions to determine which parameters are
    coefficients of variables (LHS) vs. right-hand side constants (RHS).
    Uses parentheses analysis and operator precedence to make this distinction.
    """
    lhs_params: Set[str] = set()

    # Handle bracketed terms (indexes)
    # Step 1: Extract bracketed terms from the constraint expression
    bracketed_terms = re.findall(r"\[.*?\]", constraint_expr)

    # Step 2: Replace bracketed terms with placeholders
    placeholders = {}
    modified_expr = constraint_expr
    for i, term in enumerate(bracketed_terms):
        placeholder = f" __PLACEHOLDER_{i}__ "
        placeholders.update({placeholder: term})
        modified_expr = modified_expr.replace(term, placeholder, 1)

    # Step 3: Split the modified expression around non-word characters,
    # preserving placeholders
    parts = re.split(r"(\W)", modified_expr)
    parts = [part for part in parts if part.strip() != ""]

    # Step 4: Re-insert bracketed terms in place of placeholders
    final_parts = []
    for part in parts:
        if part.startswith("__PLACEHOLDER"):
            original_term = placeholders[" " + part + " "]
            final_parts.append(original_term)
        else:
            final_parts.append(part)

    parts = final_parts

    def locate_name(
        names: List[str], parts: List[str]
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Locate indices of specified names within expression parts.

        Parameters
        ----------
        names : list[str]
            Names to search for in the expression parts.
        parts : list[str]
            Tokenized expression parts to search within.

        Returns
        -------
        dict[str, dict[str, list[int]]]
            Dictionary mapping each found name to its index positions.
        """
        dict = {}
        for name in names:
            if name in parts:
                # find the idx of param_name/var_name in parts
                name_idx = [i for i, x in enumerate(parts) if x == name]
                dict[name] = {"indexes": name_idx}

        return dict

    def in_parentheses(index: int) -> Tuple[int, int]:
        """
        Count parentheses balance at given index position.

        Parameters
        ----------
        index : int
            Position in parts list to check parentheses balance.

        Returns
        -------
        tuple of (int, int)
            Number of left and right parentheses before the index.
        """
        lbrace: int = 0
        rbrace: int = 0
        for i in range(index):
            if parts[i] == "(":
                lbrace += 1
            elif parts[i] == ")":
                rbrace += 1

        return lbrace, rbrace

    param_dict = locate_name(param_names, parts)
    var_dict = locate_name(var_names, parts)

    for param_name, param_indexes in param_dict.items():
        for param_idx in param_indexes["indexes"]:
            param_lbrace, param_rbrace = in_parentheses(param_idx)
            for var_name, var_indexes in var_dict.items():
                for var_idx in var_indexes["indexes"]:
                    var_lbrace, var_rbrace = in_parentheses(var_idx)
                    if (param_lbrace, param_rbrace) == (var_lbrace, var_rbrace):
                        # in the same parentheses
                        for i in range(
                            min(param_idx, var_idx), max(param_idx, var_idx)
                        ):
                            if parts[i] in ["+", "-", "=", "<", ">"]:
                                break
                            elif parts[i] in ["*", "/"]:
                                lhs_params.add(param_name)
                                break
                    elif (
                        abs(param_lbrace - var_lbrace) == 1
                        and param_rbrace - var_rbrace == 0
                    ):
                        # one parenthesis in between
                        for i in range(
                            min(param_idx, var_idx), max(param_idx, var_idx)
                        ):
                            if parts[i] in ["+", "-", "=", "<", ">", "("]:
                                break
                            elif parts[i] in ["*", "/"]:
                                lhs_params.add(param_name)
                                break
                    elif (
                        param_lbrace - var_lbrace == 0
                        and abs(param_rbrace - var_rbrace) == 1
                    ):
                        # one parenthesis in between
                        for i in range(
                            max(param_idx, var_idx), min(param_idx, var_idx), -1
                        ):
                            if parts[i] in ["+", "-", "=", "<", ">", ")"]:
                                break
                            elif parts[i] in ["*", "/"]:
                                lhs_params.add(param_name)
                                break

    return lhs_params


def pyomo2json(
    model: pe.ConcreteModel, termination_condition: str = "Unknown"
) -> ModelDictWithPyomo:
    """
    Convert a Pyomo optimization model to JSON representation.

    Parameters
    ----------
    model : pyomo.core.base.PyomoModel.ConcreteModel
        The Pyomo optimization model to convert.
    termination_condition : str or TerminationCondition, default="Unknown"
        The solver termination condition for the model.

    Returns
    -------
    dict
        Dictionary representation containing:
        - model_class: The original Pyomo model object
        - model_status: Solver termination condition
        - model_type: Problem type ("LP", "IP", etc.)
        - components: Organized model components (sets, parameters, variables,
          constraints, objectives) with metadata
        - Each component includes name, indexing info, descriptions, relationships

    Notes
    -----
    - Automatically detects problem type (LP/IP) based on variable domains
    - Identifies parameter roles (LHS coefficients vs RHS constants)
    - Tracks relationships between parameters/variables and constraints
    - Extracts index sets and component descriptions from doc strings
    - Handles both indexed and non-indexed components
    """
    model_dict = {}
    # model_dict["model name"] = model.name
    model_dict["model_class"] = model
    model_dict["model_status"] = termination_condition
    model_dict["model_type"] = "LP"
    model_dict["model_description"] = None

    model_dict["components"] = {}

    model_dict["components"]["sets"] = {}
    set_name: str
    _set: pe.Set
    for set_name, _set in model.component_map(pe.Set).items():
        set_dict = {}
        set_dict["name"] = set_name
        set_dict["is_indexed"] = False
        set_dict["description"] = _set.doc
        model_dict["components"]["sets"][set_name] = set_dict

    model_dict["components"]["parameters"] = {}
    param_name: str
    param: pe.Param
    for param_name, param in model.component_map(pe.Param).items():
        param_dict = {}
        param_dict["name"] = param_name
        if param.is_indexed():
            param_dict["is_indexed"] = param.is_indexed()
            param_dict["index_set"] = param.index_set()  # store the index set object
        else:
            param_dict["is_indexed"] = param.is_indexed()
            param_dict["index_set"] = None  # non_indexed_param[None] is accessible
        if param.mutable:
            param_dict["is_RHS"] = True  # revisit later
            param_dict["is_mutable"] = True
        else:
            param_dict["is_RHS"] = False  # revisit later
            param_dict["is_mutable"] = False
        param_dict["cons_in"] = set()

        param_dict["description"] = param.doc
        model_dict["components"]["parameters"][param_name] = param_dict

        # add description to default sets
        set_name = param_name + "_index"
        if set_name in model_dict["components"]["sets"]:
            model_dict["components"]["sets"][set_name][
                "description"
            ] = f"index set for {param_name} parameter"

    model_dict["components"]["variables"] = {}
    var_name: str
    var: IndexedVar
    for var_name, var in model.component_map(pe.Var).items():
        var_dict = {}
        var_dict["name"] = var_name
        if var.is_indexed():
            var_dict["is_indexed"] = var.is_indexed()
            var_dict["index_set"] = var.index_set()  # store the index set object
        else:
            var_dict["is_indexed"] = var.is_indexed()
            var_dict["index_set"] = None  # non_indexed_var[None] is accessible
        var_dict["cons_in"] = set()
        var_dict["description"] = var.doc
        model_dict["components"]["variables"][var_name] = var_dict

        # check if the model is an IP
        if model_dict["model_type"] != "IP":
            for var_idx in var:
                var_i: VarData = var[var_idx]
                if var_i.is_binary():
                    model_dict["model_type"] = "IP"

        # add description to default sets
        set_name = var_name + "_index"
        if set_name in model_dict["components"]["sets"]:
            model_dict["components"]["sets"][set_name][
                "description"
            ] = f"index set for {var_name} variable"

    model_dict["components"]["constraints"] = {}
    con_name: str
    con: IndexedConstraint
    for con_name, con in model.component_map(pe.Constraint).items():
        con_dict = {}
        con_dict["name"] = con_name
        if con.is_indexed():
            con_dict["is_indexed"] = con.is_indexed()
            con_dict["index_set"] = con.index_set()  # store the index set object
        else:
            con_dict["is_indexed"] = con.is_indexed()
            con_dict["index_set"] = None  # non_indexed_con[None] is accessible

        # for each type of constraint, identify the mutable parameter names
        # AND identify the RHS parameters
        con_dict["params_in"] = set()
        con_dict["vars_in"] = set()
        for con_idx in con:
            con_i: ConstraintData = con[con_idx]
            expr_params = identify_mutable_parameters(con_i.expr)
            expr_vars = identify_variables(con_i.expr)
            for p in expr_params:
                p_name = p.name.split("[")[0]
                con_dict["params_in"].add(p_name)
                model_dict["components"]["parameters"][p_name]["cons_in"].add(con_name)

            for v in expr_vars:
                v_name = v.name.split("[")[0]
                con_dict["vars_in"].add(v_name)
                model_dict["components"]["variables"][v_name]["cons_in"].add(con_name)

            if con_i.expr is not None:
                con_i_expr = con_i.expr.to_string()
            else:
                raise ValueError(f"Constraint {con_name} has no expression.")

            lhs_params = find_lhs_params(
                con_i_expr,
                list(model_dict["components"]["parameters"].keys()),
                list(model_dict["components"]["variables"].keys()),
            )
            for lhs_param in lhs_params:
                model_dict["components"]["parameters"][lhs_param]["is_RHS"] = False

        con_dict["description"] = con.doc
        model_dict["components"]["constraints"][con_name] = con_dict

        # add description to default sets
        set_name = con_name + "_index"
        if set_name in model_dict["components"]["sets"]:
            model_dict["components"]["sets"][set_name][
                "description"
            ] = f"index set for {con_name} constraint"

    model_dict["components"]["objective"] = {}
    obj_name: str
    obj: ScalarObjective
    for obj_name, obj in model.component_map(pe.Objective).items():
        obj_dict = {}
        obj_dict["name"] = obj_name
        if obj.sense == 1:
            obj_dict["sense"] = "minimize"
        elif obj.sense == -1:
            obj_dict["sense"] = "maximize"

        if termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
        ]:
            obj_dict["optimal_value"] = "N/A due to infeasibility"
        else:
            obj_dict["optimal_value"] = obj()
        # if termination_condition == "optimal":
        #     obj_dict["optimal_value"] = obj()
        # elif termination_condition == "maxTimeLimit":
        #     obj_dict["optimal_value"] = obj()
        # elif termination_condition == "infeasible":
        #     obj_dict["optimal_value"] = "N/A due to infeasibility"

        obj_dict["is_indexed"] = False
        obj_dict["description"] = obj.doc
        model_dict["components"]["objective"][obj_name] = obj_dict

    return model_dict  # type: ignore[return-value]


def iis2json(ilp_path: str, model_dict: ModelDictWithPyomo) -> ModelDictWithPyomo:
    """
    Extract Irreducible Infeasible Subsystem (IIS) information from ILP file.

    Parameters
    ----------
    ilp_path : str
        Path to the ILP file containing IIS information.
    model_dict : dict
        Model dictionary to be updated with IIS information.

    Returns
    -------
    None
        Modifies model_dict in place by adding "iis" key with constraint info.

    Notes
    -----
    Parses ILP file to identify constraints involved in the infeasibility
    and extracts their parameters and variables. Only processes models
    with infeasible or infeasible-or-unbounded termination conditions.
    """
    constr_names = set()
    iis_dict = {}
    if model_dict["model_status"] in [
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ]:
        with open(ilp_path, "r") as file:
            ilp_string = file.read()
        file.close()
        ilp_lines = ilp_string.split("\n")
        for iis_line in ilp_lines:
            if ":" in iis_line:
                constr_name = iis_line.split(":")[0].split("(")[0].replace(" ", "")
                constr_names.add(constr_name)

        for const_name in constr_names:
            iis_dict[const_name] = {
                "params_in": model_dict["components"]["constraints"][const_name][
                    "params_in"
                ],
                "vars_in": model_dict["components"]["constraints"][const_name][
                    "vars_in"
                ],
            }
    model_dict["iis"] = iis_dict

    iis_description = iis_translation(model_dict)
    model_dict["iis_description"] = iis_description

    return model_dict


def initial_loading(
    file: UploadedFile | str, is_uploaded: bool = True
) -> Tuple[ModelsContainer, str]:
    """
    Load and initialize a Pyomo optimization model from file.

    Parameters
    ----------
    file : UploadedFile | str
        If is_uploaded=True, a streamlit uploaded file object.
        If is_uploaded=False, a file path string.
    is_uploaded : bool, default=True
        Whether the file is uploaded via streamlit or loaded from filesystem.

    Returns
    -------
    tuple of (dict, str)
        - models_dict: Dictionary containing model representation and instance
        - code: String representation of the model code

    Notes
    -----
    - Solves the model using Gurobi solver
    - Creates IIS (Irreducible Infeasible Subsystem) files for infeasible models
    - Converts model to JSON representation with component analysis
    - Handles both uploaded files and local file paths
    - Returns complete model dictionary ready for analysis
    """
    if isinstance(file, UploadedFile) and is_uploaded:
        code = file.getvalue().decode("utf-8")
        spec = importlib.util.spec_from_loader("uploaded_model", loader=None)
        if spec is None:
            raise ImportError("Could not create a module spec for the uploaded model.")

        uploaded_model = importlib.util.module_from_spec(spec)
        sys.modules["uploaded_model"] = uploaded_model

        # Execute the code in the context of the new module
        exec(code, uploaded_model.__dict__)

        model = uploaded_model.model
        model_name = os.path.splitext(file.name)[0]

    elif isinstance(file, str) and not is_uploaded:
        with open(file, "r") as f:
            code = f.read()
        f.close()
        directory_path = os.path.dirname(file)
        model_name = os.path.splitext(os.path.basename(file))[0]
        module = importlib.import_module(directory_path + "." + model_name)
        model = module.model
    else:
        raise ValueError("Invalid file type or is_uploaded flag.")

    ilp_path = ""
    solver = SolverFactory("gurobi")
    results = solver.solve(model, tee=True)
    status = results.solver.status
    termination_condition = results.solver.termination_condition
    logger.debug(
        f"Model {model_name} loaded, "
        f"Solver Status: {status}, Termination Condition: {termination_condition}"
    )

    if termination_condition in [
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ]:
        if not os.path.exists("logs/ilps"):
            os.makedirs("logs/ilps")
        ilp_name = iis.write_iis(
            model, "logs/ilps/" + model_name + ".ilp", solver="gurobi"
        )
        ilp_path = os.path.abspath("logs/ilps/" + model_name + ".ilp")
        logger.debug("model name:", model_name)
        logger.debug(f"ilp name: {ilp_name}, ilp path: {ilp_path}")

    model_dict = pyomo2json(model, termination_condition=termination_condition)
    model_dict = iis2json(ilp_path, model_dict)
    model_dict.update({"code": code})
    models_dict = {
        "model_representation": {},
        "model_1": model_dict,
    }
    return models_dict, code


def iis_translation(model_dict: ModelDictWithPyomo) -> str:
    """
    Generate human-readable translation of IIS (Irreducible Infeasible Subsystem).

    Parameters
    ----------
    model_dict : dict
        Model dictionary containing IIS information in the "iis" key.

    Returns
    -------
    str
        Human-readable description of constraints, parameters, and variables
        involved in the infeasibility.

    Notes
    -----
    Creates natural language explanation of which constraints are causing
    infeasibility and what parameters/variables are involved, making the
    IIS information accessible to non-technical users.
    """
    iis_dict = model_dict.get("iis", {})  # Safe access for optional key
    translation = ""
    for con_name in iis_dict:
        param_names = iis_dict[con_name]["params_in"]
        var_names = iis_dict[con_name]["vars_in"]
        translation_per_con = (
            f"Constraints {con_name} are in the IIS, with the following parameters: "
        )
        for i, param_name in enumerate(param_names):
            if i == len(param_names) - 1:
                translation_per_con += (
                    f"{param_name}; and with the following variables: "
                )
            else:
                translation_per_con += f"{param_name}, "
        for i, var_name in enumerate(var_names):
            if i == len(var_names) - 1:
                translation_per_con += f"{var_name}. \n"
            else:
                translation_per_con += f"{var_name}, "
        translation += translation_per_con
    return translation


def update_model_representation(
    models_dict: ModelsContainer, model_name: str = "model_1"
) -> None:
    """
    Update model representation dictionary by copying from specified model.

    Parameters
    ----------
    models_dict : dict
        Dictionary containing multiple model representations.
    model_name : str, default="model_1"
        Key identifying which model to use as reference for updating.

    Returns
    -------
    None
        Modifies models_dict in place by creating/updating "model_representation" key.

    Notes
    -----
    Creates a cleaned version of the model dictionary by copying all components
    except index_set objects, which are not serializable. Used to prepare
    model data for JSON serialization and agent processing.
    """
    models_dict["model_representation"] = {}  # type: ignore[assignment]
    model_representation = models_dict["model_representation"]
    ref_model_dict = models_dict[model_name]
    # exclude model class
    model_representation["code"] = ref_model_dict["code"]
    model_representation["model_status"] = ref_model_dict["model_status"]
    model_representation["model_type"] = ref_model_dict["model_type"]
    model_representation["model_description"] = ref_model_dict["model_description"]
    model_representation["components"] = {}
    component_types = ["sets", "parameters", "variables", "constraints", "objective"]
    # copy everything except index_set
    for component_type in component_types:
        model_representation["components"][component_type] = {}
        for component_name, component_info in ref_model_dict["components"][
            component_type
        ].items():
            model_representation["components"][component_type][component_name] = {}
            for key, value in component_info.items():
                if key != "index_set":
                    model_representation["components"][component_type][component_name][
                        key
                    ] = value

    if "iis" in ref_model_dict:
        model_representation["iis"] = ref_model_dict["iis"]

    if "iis_description" in ref_model_dict:
        model_representation["iis_description"] = ref_model_dict["iis_description"]


def extract_component_descriptions(
    models_dict: ModelsContainer,
) -> ComponentsContainerSerializable:
    """
    Extract component descriptions from model representation dictionary.

    Parameters
    ----------
    models_dict : dict
        Dictionary containing model representation with component information.

    Returns
    -------
    dict
        Deep copy of component descriptions organized by component type
        (sets, parameters, variables, constraints, objectives).

    Notes
    -----
    Used to provide component descriptions to agents for natural language
    understanding of the optimization model structure and semantics.
    """
    ref_model_dict = models_dict["model_representation"]["components"]
    component_descriptions = copy.deepcopy(ref_model_dict)
    return component_descriptions  # type: ignore[return-value]


def replace(src_code: str, old_code: str, new_code: str) -> str:
    """
    TAKEN FROM AUTOGEN: https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat_optiguide/  # noqa: E501
    Inserts new code into the source code by replacing a specified old
    code block.

    Args:
        src_code (str): The source code to modify.
        old_code (str): The code block to be replaced.
        new_code (str): The new code block to insert.

    Returns:
        str: The modified source code with the new code inserted.

    Raises:
        None

    Example:
        src_code = 'def hello_world():\n    # CODE GOES HERE'
        old_code = '# CODE GOES HERE'
        new_code = 'print("Bonjour, monde!")\nprint("Hola, mundo!")'
        modified_code = _replace(src_code, old_code, new_code)
        print(modified_code)
        # Output:
        # def hello_world():
        #     print("Bonjour, monde!")
        #     print("Hola, mundo!")
    """
    pattern = r"( *){old_code}".format(old_code=old_code)
    match = re.search(pattern, src_code, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Pattern '{old_code}' not found in source code")

    head_spaces = match.group(1)
    new_code = "\n".join([head_spaces + line for line in new_code.split("\n")])
    rst = re.sub(pattern, new_code, src_code)
    return rst


def insert_code(src_code: str, new_lines: str, code_type: str) -> str:
    """
    Insert code patch into source code at designated location.

    Adapted from AUTOGEN: https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat_optiguide/  # noqa: E501

    Parameters
    ----------
    src_code : str
        The original source code to modify.
    new_lines : str
        The new code to insert into the source code.
    code_type : str
        Type identifier for the code insertion (currently unused).

    Returns
    -------
    str
        Modified source code with new_lines inserted at the designated location.

    Notes
    -----
    Currently replaces "# YOUR CODE GOES HERE" placeholder with the new code.
    The code_type parameter is available for future extensibility.
    """
    # # # for now, we have # OPTICHAT REVISION CODE GOES HERE and # OPTICHAT PRINT CODE GOES HERE  # noqa: E501
    # # return replace(src_code, '# CODE GOES HERE', new_lines)
    # if code_type == 'REVISION':
    #     return replace(src_code, f"# OPTICHAT {code_type} CODE GOES HERE", new_lines)
    # elif code_type == 'PRINT':
    #     return replace(src_code, f"# OPTICHAT {code_type} CODE GOES HERE", new_lines)
    # else:
    #     raise ValueError(f"Invalid code type: {code_type}")
    return replace(src_code, "# YOUR CODE GOES HERE", new_lines)


def run_with_exec(src_code: str) -> str:
    """
    Execute Python source code and capture both output and exceptions.

    Parameters
    ----------
    src_code : str
        Python source code to execute.

    Returns
    -------
    str
        Captured stdout output from execution, plus traceback if exception occurred.

    Notes
    -----
    Executes code in isolated local namespace and redirects stdout to capture
    print statements and other output. If an exception occurs, includes the
    full traceback in the returned string. Used for safe code execution
    during model analysis and debugging.
    """
    locals_dict = {}
    output = io.StringIO()

    try:
        with redirect_stdout(output):
            exec(src_code, locals_dict, locals_dict)
        return output.getvalue()
    except Exception:
        import traceback

        return output.getvalue() + "\n" + traceback.format_exc()


def var_in_con(constraint_expr: Any) -> List[Any]:
    """
    Extract variables from a constraint expression.

    Parameters
    ----------
    constraint_expr : pyomo expression
        The constraint expression to analyze.

    Returns
    -------
    list
        List of variables found in the constraint expression.

    Notes
    -----
    Uses Pyomo's identify_variables function to find all variables
    present in the given constraint expression.
    """
    vars_list = list(identify_variables(constraint_expr))
    return vars_list


def param_in_con(constraint_expr: Any) -> List[Any]:
    """
    Extract parameters from a constraint expression.

    Parameters
    ----------
    constraint_expr : pyomo expression
        The constraint expression to analyze.

    Returns
    -------
    list
        List of mutable parameters found in the constraint expression.

    Notes
    -----
    Uses Pyomo's identify_mutable_parameters function to find all mutable
    parameters present in the given constraint expression.
    """
    params_list = list(identify_mutable_parameters(constraint_expr))
    return params_list


def get_files_generator(folder_name: str) -> Generator[str, None, None]:
    """
    Generate file paths for all Python files in a folder.

    Parameters
    ----------
    folder_name : str
        Name of the folder to search for Python files.

    Yields
    ------
    str
        Full path to each Python file found in the folder.

    Examples
    --------
    >>> list(get_files_generator("video_showcase"))
    ['video_showcase/pdi_inf_1.py', 'video_showcase/pdi_inf_2.py']

    Notes
    -----
    Only returns files with .py extension. Uses generator pattern
    for memory efficiency when dealing with large directories.
    """
    files_and_dirs = os.listdir(folder_name)
    for f in files_and_dirs:
        if os.path.isfile(os.path.join(folder_name, f)) and f.endswith(".py"):
            yield os.path.join(folder_name, f)


def get_files(folder_name: str) -> Tuple[List[str], List[str]]:
    """
    Get all Python files in a folder, separated by feasible/infeasible.

    Parameters
    ----------
    folder_name : str
        Name of the folder to search for Python files.

    Returns
    -------
    tuple of (list, list)
        - infeasible_files: List of paths to infeasible model files (contain "_inf_")
        - feasible_files: List of paths to feasible model files (no "_inf_")

    Examples
    --------
    >>> infeas, feas = get_files("video_showcase")
    >>> print(infeas)
    ['video_showcase/pdi_inf_1.py', 'video_showcase/pdi_inf_2.py']
    >>> print(feas)
    ['video_showcase/pdi.py', 'video_showcase/other.py']

    Notes
    -----
    Categorizes files based on "_inf_" pattern in filename to distinguish
    between feasible and infeasible optimization model examples.
    """
    files_and_dirs = os.listdir(folder_name)
    infeasible_files = []
    feasible_files = []
    for f in files_and_dirs:
        if os.path.isfile(os.path.join(folder_name, f)) and f.endswith(".py"):
            if "_inf_" in f:
                infeasible_files.append(os.path.join(folder_name, f))
            else:
                feasible_files.append(os.path.join(folder_name, f))

    return infeasible_files, feasible_files


def get_skipJSON(model_representation: ModelDictSerializable) -> Dict[str, Any]:
    """
    Extract model and component descriptions for quick loading.

    Parameters
    ----------
    model_representation : dict
        Dictionary containing complete model representation with descriptions.

    Returns
    -------
    dict
        Simplified JSON containing only model description and component
        descriptions, used to skip the interpretation process.

    Notes
    -----
    Creates a lightweight JSON structure that helps skip the time-consuming
    model interpretation process when component descriptions are already
    available from previous analysis.
    """
    COMPONENT_TYPES = ["sets", "parameters", "variables", "constraints", "objective"]
    skipJSON = {
        "model_description": model_representation["model_description"],
        "components": {
            component_type: {
                component_name: component_dict["description"]
                for component_name, component_dict in model_representation[
                    "components"
                ][component_type].items()
            }
            for component_type in COMPONENT_TYPES
        },
    }
    return skipJSON


def feed_skipJSON(
    skipJSON: Dict[str, Any],
    models_dict: ModelsContainer,
    queried_model: str = "model_1",
) -> ModelsContainer:
    """
    Load pre-computed descriptions into model dictionary.

    Parameters
    ----------
    skipJSON : dict
        Dictionary containing model and component descriptions from get_skipJSON().
    models_dict : ModelsContainer
        Dictionary containing model representations to be updated.
    queried_model : str, default="model_1"
        Key identifying which model in models_dict to update.

    Returns
    -------
    dict
        Updated models_dict with descriptions loaded from skipJSON.

    Notes
    -----
    Populates model dictionary with pre-computed descriptions to avoid
    re-running the interpretation process. Used for faster loading when
    component descriptions are already available.
    """
    COMPONENT_TYPES = ["sets", "parameters", "variables", "constraints", "objective"]
    model_dict = models_dict[queried_model]
    model_dict["model_description"] = skipJSON["model_description"]
    for component_type in COMPONENT_TYPES:
        for component_name, component_dict in model_dict["components"][
            component_type
        ].items():
            component_dict["description"] = skipJSON["components"][component_type][
                component_name
            ]

    return models_dict
