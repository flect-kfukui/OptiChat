import copy
import json
import re
import time
from typing import Any, List, Optional, Tuple

from loguru import logger
from openai import Client, OpenAI, Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatText

from extractor import extract_component_descriptions, run_with_exec
from internal_tools import (
    components_retrival,
    evaluate_modification,
    feasibility_restoration,
    fnArgsDecoder,
    sensitivity_analysis,
    syntax_guidance,
)
from optichat_types import (
    DecisionDict,
    ModelDictWithPyomo,
    ModelsContainer,
    TeamConversationMessage,
)
from prompts import get_prompts

# import streamlit as st


class Agent:
    """
    Base class for AI agents in the optimization model analysis system.

    This class provides common functionality for different types of agents
    that interact with optimization models and provide analysis capabilities.

    Parameters
    ----------
    name : str
        The name identifier for the agent.
    description : str
        A description of the agent's purpose and capabilities.
    client : Client or OpenAI
        The OpenAI client instance for making API calls.
    llm : str, default="gpt-4-turbo-preview"
        The language model to use for completions.
    **kwargs : dict
        Additional configuration parameters including function names, tools, etc.

    Attributes
    ----------
    system_prompt : str
        The system prompt used for LLM interactions.
    function_names : list, optional
        List of available function names for the agent.
    tools : list, optional
        Available tools for the agent.
    team_conversation_filename : str
        Path to the team conversation log file.
    chat_history_filename : str
        Path to the detailed chat history log file.
    """

    def __init__(
        self,
        name: str,
        description: str,
        client: OpenAI | Client,
        llm: str = "gpt-4-turbo-preview",
        **kwargs,
    ):
        """
        Initialize an AI agent with configuration and LLM client.

        Parameters
        ----------
        name : str
            The name identifier for the agent.
        description : str
            A description of the agent's purpose and capabilities.
        client : Client or OpenAI
            The OpenAI client instance for making API calls.
        llm : str, default="gpt-4-turbo-preview"
            The language model to use for completions.
        **kwargs : dict
            Additional configuration parameters including:
            - function_names : list, optional - Available function names
            - tools : list, optional - Available tools for the agent
            - multiple_tools : list, optional - Tools for multiple index specs
            - single_tools : list, optional - Tools for single index specs
            - none_tools : list, optional - Tools for non-indexed components
            - all_tools : list, optional - Combined tools for all modes
            - tool_choice : str, optional - Default tool choice setting
            - syntax_guidance_tool : list, optional - Syntax guidance tools

        Notes
        -----
        Sets up the base configuration for all AI agents in the system.
        Initializes tool configurations, file paths for logging, and
        establishes the connection to the language model client.
        """
        self.name: str = name
        self.description: str = description
        self.client: OpenAI | Client = client
        self.system_prompt: str = "You're a helpful assistant."
        self.kwargs = kwargs
        self.llm: str = llm

        self.function_names: str | None = kwargs.get("function_names", None)
        self.tools = kwargs.get("tools", None)
        self.multiple_tools = kwargs.get("multiple_tools", None)
        self.single_tools = kwargs.get("single_tools", None)
        self.none_tools = kwargs.get("none_tools", None)
        self.all_tools = kwargs.get("all_tools", None)
        self.tool_choice = kwargs.get("tool_choice", None)
        self.syntax_guidance_tool = kwargs.get("syntax_guidance_tool", None)

        self.team_conversation_filename: str = "./logs/team_conversation.txt"
        self.chat_history_filename: str = "./logs/detailed_chat_history.txt"

    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        seed: int = 10,
        stream: bool = False,
    ) -> Stream[ChatCompletionChunk] | str | None:
        """
        Make a call to the language model with either a prompt or messages.

        Parameters
        ----------
        prompt : str, optional
            Single prompt string to send to the LLM. Mutually exclusive with messages.
        messages : list of dict, optional
            List of message dictionaries with 'role' and 'content' keys.
            Mutually exclusive with prompt.
        seed : int, default=10
            Random seed for reproducible results.
        stream : bool, default=False
            Whether to stream the response.

        Returns
        -------
        Stream[ChatCompletionChunk] | str | None
            The LLM response content if stream=False, otherwise completion object.

        Notes
        -----
        Exactly one of prompt or messages must be provided. If prompt is provided,
        it will be formatted as a user message with the system prompt.
        """
        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)

        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if prompt is not None:
            messages = [
                ChatCompletionSystemMessageParam(
                    {"role": "system", "content": self.system_prompt}
                ),
                ChatCompletionUserMessageParam({"role": "user", "content": prompt}),
            ]

        # print("=" * 10)
        # print(f'llm_call is called, the following messages are sent to the llm: ')
        # for message in messages:
        #     print(f'{message["role"]}: {message["content"]}')
        # print("=" * 10)

        if isinstance(self.client, (OpenAI, Client)):
            completion: ChatCompletion | Stream[ChatCompletionChunk] = (
                self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,  # type: ignore
                    seed=seed,
                    stream=stream,
                )
            )  # type: ignore

            if stream and isinstance(completion, Stream):
                return completion
            elif isinstance(completion, ChatCompletion):
                content = completion.choices[0].message.content
                return content
            else:
                raise ValueError(
                    f"Unexpected completion type received from LLM.: {completion}"
                )

    @staticmethod
    def generate_pseudo_messages(
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
        new_prompt: str,
    ) -> List[ChatCompletionMessageParam]:
        """
        Generate pseudo messages for agent interaction.

        Parameters
        ----------
        messages : list[ChatCompletionMessageParam]
            Original message history.
        team_conversation : list[TeamConversationMessage]
            Team conversation history with agent responses.
        new_prompt : str
            New prompt to add to the conversation.

        Returns
        -------
        list[ChatCompletionMessageParam]
            Modified message list with team conversation context and new prompt.

        Notes
        -----
        Integrates team conversation history as system messages and adds
        the new prompt as a user message to create context for agent interactions.
        """
        pseudo_messages: list[ChatCompletionMessageParam] = copy.deepcopy(messages)
        if team_conversation:
            for message in team_conversation:
                if message["agent_name"] in ["Syntax reminder", "Code reminder"]:
                    pseudo_messages.append(
                        ChatCompletionSystemMessageParam(
                            {
                                "role": "system",
                                "content": f'{message["agent_name"]}: \n\n'
                                + message["agent_response"],
                            }
                        )
                    )
                else:
                    pseudo_messages.append(
                        ChatCompletionAssistantMessageParam(
                            {
                                "role": "assistant",
                                "content": (
                                    f'I am {message["agent_name"]} in Assistant Team. '
                                    + "\n\n"
                                    + message["agent_response"]
                                ),
                            }
                        )
                    )

        pseudo_messages.append(
            ChatCompletionUserMessageParam({"role": "user", "content": new_prompt})
        )
        return pseudo_messages

    def save_team_conversation(
        self, team_conversation: List[TeamConversationMessage]
    ) -> None:
        """
        Save team conversation to a file for debugging and analysis.

        Parameters
        ----------
        team_conversation : list[TeamConversationMessage]
            List of conversation messages from different agents.
            Each dict should contain 'agent_name' and 'agent_response' keys.

        Notes
        -----
        Appends each message to the team conversation file with proper formatting.
        Used for tracking multi-agent interactions during the optimization process.
        """
        with open(self.team_conversation_filename, "a") as f:
            for message in team_conversation:
                f.write(f"{message['agent_name']}: {message['agent_response']}\n\n")

    def print_in_and_out(
        self, prompt: str, llm_response: str, agent_name: Optional[str] = None
    ) -> None:
        """
        Print formatted input prompt and LLM response for debugging.

        Parameters
        ----------
        prompt : str
            The input prompt sent to the LLM.
        llm_response : str
            The response received from the LLM.
        agent_name : str, optional
            Name of the agent for the header. If None, uses self.name.

        Notes
        -----
        Provides structured output for debugging agent conversations and
        monitoring LLM interactions during optimization analysis.
        """
        if agent_name is None:
            agent_name = self.name

        logger.debug("=" * 5 + str(agent_name) + "=" * 5)
        logger.debug("-" * 5 + "prompt:" + "-" * 5)
        logger.debug(prompt)
        logger.debug("-" * 5 + "llm_response:" + "-" * 5)
        logger.debug(llm_response)

    def llm_call_exp(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        seed: int = 10,
        temperature: float = 0.1,
        json_mode: bool = False,
        stream: bool = False,
    ) -> Stream[ChatCompletionChunk] | str | None:
        """
        Extended LLM call with additional parameters for temperature and JSON mode.

        Parameters
        ----------
        prompt : str, optional
            Single prompt string to send to the LLM.
        messages : list[ChatCompletionMessageParam] | None
            List of message dictionaries with 'role' and 'content' keys.
        seed : int, default=10
            Random seed for reproducible results.
        temperature : float, default=0.1
            Sampling temperature for response randomness.
        json_mode : bool, default=False
            Whether to request JSON-formatted responses.
        stream : bool, default=False
            Whether to stream the response.

        Returns
        -------
        Stream[ChatCompletionChunk] | str | None
            The LLM response content or completion object for streaming.

        Notes
        -----
        Similar to llm_call but with additional control over temperature and
        response format. Handles both regular OpenAI models and special cases like o3.
        """
        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)

        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if prompt is not None:
            messages = [
                ChatCompletionSystemMessageParam(
                    {"role": "system", "content": self.system_prompt}
                ),
                ChatCompletionUserMessageParam({"role": "user", "content": prompt}),
            ]

        if json_mode:
            response_format = ResponseFormatJSONObject({"type": "json_object"})
        else:
            response_format = ResponseFormatText({"type": "text"})

        if type(self.client) in [OpenAI, Client]:
            if self.llm not in ["o3"]:
                completion: ChatCompletion | Stream[ChatCompletionChunk] = (
                    self.client.chat.completions.create(
                        model=self.llm,
                        messages=messages,  # type: ignore
                        seed=seed,
                        temperature=temperature,
                        response_format=response_format,
                        stream=stream,
                    )  # type: ignore
                )
            else:
                completion: ChatCompletion | Stream[ChatCompletionChunk] = (
                    self.client.chat.completions.create(
                        model=self.llm,
                        messages=messages,  # type: ignore
                        seed=seed,
                        response_format=response_format,
                        stream=stream,
                    )  # type: ignore
                )

            if stream and isinstance(completion, Stream):
                return completion
            elif isinstance(completion, ChatCompletion):
                content = completion.choices[0].message.content
                return content
            else:
                raise ValueError(
                    f"Unexpected completion type received from LLM.: {completion}"
                )


class Interpreter(Agent):
    def __init__(self, client: Client, **kwargs):
        """Initialize an Engineer agent specialized in technical optimization feedback.

        This constructor creates an agent focused on providing technical
        feedback for optimization-related queries, particularly when the
        user\'s question involves scenarios that differ from the current model.
        The Engineer agent executes tools and functions when direct interaction
        with optimization models is required.

        Parameters
        ----------
        client : Client
            The client interface for communication with the AI model.
        **kwargs
            Additional keyword arguments passed to the parent Agent class.
            See Agent.__init__ for supported parameters.

        Notes
        -----
        The Engineer agent is designed to provide technical rather than
        natural-language explanations, making it suitable for users who
        need detailed technical feedback about optimization scenarios.
        """
        """
        Initialize the Interpreter agent for optimization model analysis.

        Parameters
        ----------
        client : Client
            OpenAI client instance for making API calls.
        **kwargs : dict
            Additional configuration parameters passed to parent Agent class.

        Notes
        -----
        Specialized agent that interprets optimization models and translates
        technical concepts into natural language explanations for non-experts.
        Initializes prompt templates for model interpretation, illustration,
        and inference tasks.
        """
        super().__init__(
            name="Interpreter",
            description="This is an operations research agent that is an expert in interpreting optimization models and codes to non-experts.",
            client=client,
            **kwargs,
        )

        self._init_prompt_template()

    def _init_prompt_template(self):
        """
        Initialize prompt templates for model interpretation tasks.

        Notes
        -----
        Sets up templates for model interpretation, component description,
        illustration, and inference prompts used by the Interpreter agent.
        """
        self.interpretation_prompt_template: str = get_prompts(
            "model_interpretation_prompt"
        )  # type: ignore
        self.need2describe_prompt_template: str = get_prompts(  # type: ignore
            "need2describe_prompt"
        )
        self.interpretation_json_template: dict[str, Any] = get_prompts(
            "model_interpretation_json"
        )  # type: ignore

        self.illustration_prompt_template: str = get_prompts(
            "model_illustration_prompt"
        )  # type: ignore
        self.inference_prompt_template: str = get_prompts(  # type: ignore
            "model_inference_prompt"
        )

    def _cat(
        self, cat_need2describe: str, component_names: List[str], component_type: str
    ) -> str:
        """
        Concatenate component description request to prompt.

        Parameters
        ----------
        cat_need2describe : str
            Current accumulated description prompt.
        component_names : list
            Names of components that need description.
        component_type : str
            Type of components (e.g., 'parameters', 'variables').

        Returns
        -------
        str
            Updated prompt with component description request.
        """
        return cat_need2describe + self.need2describe_prompt_template.format(
            component_type=component_type, component_names=component_names
        )

    def _cut(self, component_type: str) -> None:
        """
        Remove a component type from the interpretation JSON template.

        Parameters
        ----------
        component_type : str
            Type of component to remove (e.g., 'parameters', 'variables').

        Notes
        -----
        Used to customize the interpretation template by removing
        component types that don't need description.
        """
        if component_type in self.interpretation_json_template["components"]:
            del self.interpretation_json_template["components"][component_type]

    def generate_interpretation(
        self, models_dict: ModelsContainer, code: str, model_name="model_1"
    ) -> ModelsContainer:
        """
        Generate model interpretation by analyzing components and adding descriptions.

        Parameters
        ----------
        models_dict : ModelsContainer
            Dictionary containing model representations and components.
        code : str
            Source code of the optimization model.
        model_name : str, default="model_1"
            Key identifying which model to interpret.

        Returns
        -------
        ModelsContainer
            Updated models_dict with component descriptions filled in.

        Raises
        ------
        Exception
            If JSON parsing fails repeatedly after 3 attempts.

        Notes
        -----
        Iteratively processes model components to generate natural language
        descriptions. Uses retry logic with 3 attempts for robustness against
        JSON parsing errors.
        """
        task_complete: bool = False
        cnt: int = 3
        while not task_complete and cnt > 0:
            self._init_prompt_template()
            cat_need2describe_prompt = ""
            need2describe = {}
            for component_type in [
                "sets",
                "parameters",
                "variables",
                "constraints",
                "objective",
            ]:
                need2describe[component_type] = []
                for key, value in models_dict[model_name]["components"][
                    component_type
                ].items():
                    if value.get("description") in ["None", None]:
                        need2describe[component_type].append(key)

                # if there are components that haven't been described, add them to the prompt
                if len(need2describe[component_type]) > 0:
                    cat_need2describe_prompt = self._cat(
                        cat_need2describe_prompt,
                        need2describe[component_type],
                        component_type,
                    )
                else:
                    self._cut(component_type)

                # print('===' * 10)
                # print('cat_need2describe_prompt:', cat_need2describe_prompt)
                # print(f'interpretation_json_template: {self.interpretation_json_template}')

            if len(cat_need2describe_prompt) > 0:
                model_interpretation_json = json.dumps(
                    self.interpretation_json_template, indent=4
                )
                # create complete prompt with components that haven't been described only
                prompt = self.interpretation_prompt_template.format(
                    code=code,
                    cat_need2describe_prompt=cat_need2describe_prompt,
                    model_interpretation_json=model_interpretation_json,
                )
            else:
                # if all the components in all the component types have been described, then no need to call interpreter
                return models_dict

            cnt -= 1
            try:
                interpretation_json: str = self.llm_call(
                    prompt=prompt, seed=cnt, stream=False
                )  # type: ignore
                logger.debug("=" * 10)
                logger.debug(f"generate_interpretation... cnt left = {cnt}/3")
                logger.debug(interpretation_json)
                logger.debug("=" * 10)
                output = interpretation_json
                # delete until the first '```json'
                if "```json" in output:
                    output = output[output.find("```json") + 7 :]
                    output = output[: output.rfind("```")]

                start = output.find("{")
                end = output.rfind("}")
                output = output[start : end + 1]

                update = json.loads(output)

                task_complete: bool = (
                    True  # mark as complete first, if any component incorrect, mark as incomplete
                )
                for key in update["components"]:
                    logger.debug(f"Interpreting {key}")
                    for component in update["components"][key]:
                        logger.debug(f"component: {component}")
                        # update models_dict with the new descriptions if format is correct,
                        # next time less components will be included in the prompt
                        if ("name" in component) and ("description" in component):
                            models_dict[model_name]["components"][key][
                                component["name"]
                            ]["description"] = component["description"]
                        else:
                            logger.debug(
                                f"Invalid component format marked!, {component}"
                            )
                            task_complete = False

            except Exception as e:
                import traceback

                logger.error(traceback.format_exc())
                logger.error("=" * 10)
                logger.error(f"generate_interpretation error... cnt left = {cnt}/3")
                logger.error(e)
                logger.error("=" * 10)
                logger.error("generate_interpretation prompt that caused the error: ")
                logger.error(prompt)
                logger.error("=" * 10)
                # logger.error(interpretation_json)
                logger.error("=" * 10)
                logger.error(f"Invalid json format!\n{e}\n Try again ...")

        if cnt == 0:
            raise Exception("Invalid json format, Failed 3 times!")

        return models_dict

    def generate_illustration(
        self, model_representation: ModelDictWithPyomo
    ) -> Stream[ChatCompletionChunk]:
        """
        Generate a natural language illustration of the optimization model.

        Parameters
        ----------
        model_representation : ModelDictWithPyomo
            Complete model representation with component descriptions.

        Returns
        -------
        Stream[ChatCompletionChunk]
            Streaming LLM response containing model illustration and explanation.

        Notes
        -----
        Creates user-friendly explanations of the optimization model structure,
        objectives, constraints, and variables for non-technical audiences.
        """
        prompt = self.illustration_prompt_template.format(
            json_representation=model_representation
        )
        logger.debug("=" * 10)
        logger.debug("generate_illustration... ")
        logger.debug("=" * 10)
        stream: Stream[ChatCompletionChunk] = self.llm_call(  # type: ignore
            prompt=prompt, stream=True
        )
        return stream

    def generate_inference(
        self, model_representation: ModelDictWithPyomo
    ) -> Stream[ChatCompletionChunk]:
        """
        Generate inference about model infeasibility using IIS information.

        Parameters
        ----------
        model_representation : ModelDictWithPyomo
            Complete model representation containing IIS information.

        Returns
        -------
        Stream[ChatCompletionChunk]
            Streaming LLM response with inference about infeasibility causes.

        Notes
        -----
        Analyzes the Irreducible Infeasible Subsystem (IIS) to provide
        insights about why the model is infeasible and what might be done
        to resolve the infeasibility.
        """

        def split_representation(representation):
            """
            Split model representation into IIS info and reduced representation.

            Parameters
            ----------
            representation : dict
                Complete model representation with IIS information.

            Returns
            -------
            tuple of (str, dict)
                IIS description string and model representation without IIS data.
            """
            # just split session_state.models_dict["model_representation"] into two parts
            reduced_json_representation = copy.deepcopy(representation)
            del reduced_json_representation["iis"]
            del reduced_json_representation["iis_description"]
            return representation["iis_description"], reduced_json_representation

        iis_info, reduced_model_representation = split_representation(
            model_representation
        )
        prompt = self.inference_prompt_template.format(
            iis_info=iis_info, json_representation=reduced_model_representation
        )
        logger.debug("=" * 10)
        logger.debug("generate_inference... ")
        logger.debug("=" * 10)
        stream: Stream[ChatCompletionChunk] = self.llm_call(  # type: ignore
            prompt=prompt, stream=True
        )

        return stream

    def generate_interpretation_exp(
        self, args, models_dict: ModelsContainer, code: str, model_name="model_1"
    ) -> Tuple[ModelsContainer, int, bool]:
        """
        Generate model interpretation with experimental settings and retry logic.

        Parameters
        ----------
        args : object
            Configuration arguments including temperature and streaming settings.
        models_dict : ModelsContainer
            Dictionary containing multiple model representations.
        code : str
            Source code of the optimization model.
        model_name : str, default="model_1"
            Key identifying which model to interpret.

        Returns
        -------
        Tuple[ModelsContainer, int, bool]
            Updated models_dict, retry count remaining, and success flag.
            Returns None if all retries failed.

        Notes
        -----
        Extended version of generate_interpretation with experimental parameters
        and enhanced error handling. Includes retry logic for robustness.
        """
        task_complete: bool = False
        cnt: int = 3
        while not task_complete and cnt > 0:
            self._init_prompt_template()
            cat_need2describe_prompt = ""
            need2describe = {}
            for component_type in [
                "sets",
                "parameters",
                "variables",
                "constraints",
                "objective",
            ]:
                need2describe[component_type] = []
                for key, value in models_dict[model_name]["components"][
                    component_type
                ].items():
                    if value.get("description") in ["None", None]:
                        need2describe[component_type].append(key)
                # if there are components that haven't been described, add them to the prompt
                if len(need2describe[component_type]) > 0:
                    cat_need2describe_prompt = self._cat(
                        cat_need2describe_prompt,
                        need2describe[component_type],
                        component_type,
                    )
                else:
                    self._cut(component_type)

            if len(cat_need2describe_prompt) > 0:
                model_interpretation_json = json.dumps(
                    self.interpretation_json_template, indent=4
                )
                # create complete prompt with components that haven't been described only
                prompt = self.interpretation_prompt_template.format(
                    code=code,
                    cat_need2describe_prompt=cat_need2describe_prompt,
                    model_interpretation_json=model_interpretation_json,
                )
            else:
                # if all the components in all the component types have been described, then no need to call interpreter
                task_complete = True
                return models_dict, cnt, task_complete

            cnt -= 1
            try:
                interpretation_json: str = self.llm_call_exp(
                    prompt=prompt,
                    seed=cnt,
                    temperature=args.temperature,
                    json_mode=args.json_mode,
                    stream=False,
                )  # type: ignore
                logger.debug("=" * 10)
                logger.debug(f"generate_interpretation... cnt left = {cnt}/3")
                logger.debug(interpretation_json)
                logger.debug("=" * 10)
                output = interpretation_json

                # print("=" * 10 + 'debug: for testing json mode only' + "=" * 10)
                # print(output)
                # print("=" * 10)

                # delete until the first '```json'
                if "```json" in output:
                    output = output[output.find("```json") + 7 :]
                    output = output[: output.rfind("```")]

                start = output.find("{")
                end = output.rfind("}")
                output = output[start : end + 1]

                update = json.loads(output)

                task_complete = True  # mark as complete first, if any component incorrect, mark as incomplete
                for key in update["components"]:
                    logger.debug(f"Interpreting {key}")
                    for component in update["components"][key]:
                        logger.debug(f"component: {component}")
                        # update models_dict with the new descriptions if format is correct,
                        # next time less components will be included in the prompt
                        if ("name" in component) and ("description" in component):
                            models_dict[model_name]["components"][key][
                                component["name"]
                            ]["description"] = component["description"]
                        else:
                            logger.debug(
                                f"Invalid component format marked!, {component}"
                            )
                            task_complete = False

            except Exception as e:
                import traceback

                logger.error(traceback.format_exc())
                logger.error("=" * 10)
                logger.error(f"generate_interpretation error... cnt left = {cnt}/3")
                logger.error(e)
                logger.error("=" * 10)
                logger.error(f"Invalid json format!\n{e}\n Try again ...")

        if cnt == 0:
            return models_dict, cnt, task_complete

        return models_dict, cnt, task_complete

    def generate_illustration_exp(
        self, args, model_representation: ModelDictWithPyomo
    ) -> Stream[ChatCompletionChunk] | str:
        """
        Generate model illustration with experimental settings.

        Parameters
        ----------
        args : object
            Configuration arguments with temperature and streaming settings.
        model_representation : ModelDictWithPyomo
            Complete model representation with component descriptions.

        Returns
        -------
        Stream[ChatCompletionChunk] | str
            LLM response with model illustration, streamed or complete.

        Notes
        -----
        Extended version of generate_illustration with configurable temperature
        and streaming options for experimental analysis workflows.
        """
        prompt = self.illustration_prompt_template.format(
            json_representation=model_representation
        )
        logger.debug("=" * 10)
        logger.debug("generate_illustration... ")
        logger.debug("=" * 10)
        stream_or_completion: Stream[ChatCompletionChunk] | str | None = (
            self.llm_call_exp(
                prompt=prompt,
                temperature=args.temperature,
                stream=args.illustration_stream,
            )
        )
        assert stream_or_completion is not None
        return stream_or_completion

    def generate_inference_exp(
        self, args, model_representation: ModelDictWithPyomo
    ) -> Stream[ChatCompletionChunk] | str:
        """
        Generate inference about model infeasibility with experimental settings.

        Parameters
        ----------
        args : object
            Configuration arguments with temperature and streaming settings.
        model_representation : ModelDictWithPyomo
            Complete model representation containing IIS information.

        Returns
        -------
        str or completion object
            LLM response with infeasibility inference, streamed or complete.

        Notes
        -----
        Extended version of generate_inference with configurable temperature
        and streaming options for experimental infeasibility analysis.
        """

        def split_representation(representation):
            """
            Split model representation into IIS info and reduced representation.

            Parameters
            ----------
            representation : dict
                Complete model representation with IIS information.

            Returns
            -------
            tuple of (str, dict)
                IIS description string and model representation without IIS data.
            """
            # just split session_state.models_dict["model_representation"] into two parts
            reduced_json_representation = copy.deepcopy(representation)
            del reduced_json_representation["iis"]
            del reduced_json_representation["iis_description"]
            return representation["iis_description"], reduced_json_representation

        iis_info, reduced_model_representation = split_representation(
            model_representation
        )
        prompt = self.inference_prompt_template.format(
            iis_info=iis_info, json_representation=reduced_model_representation
        )
        logger.debug("=" * 10)
        logger.debug("generate_inference... ")
        logger.debug("=" * 10)
        stream_or_completion: Stream[ChatCompletionChunk] | str | None = (
            self.llm_call_exp(
                prompt=prompt,
                temperature=args.temperature,
                stream=args.inference_stream,
            )
        )
        assert stream_or_completion is not None
        return stream_or_completion


class Coordinator(Agent):
    def __init__(
        self, client: Client, agents: List[Agent], max_rounds: int = 5, **kwargs
    ):
        """
        Initialize the Coordinator agent for multi-agent task management.

        Parameters
        ----------
        client : Client
            OpenAI client instance for making API calls.
        agents : list of Agent
            List of available agents that can be coordinated and assigned tasks.
        max_rounds : int, default=5
            Maximum number of coordination rounds before termination.
        **kwargs : dict
            Additional configuration parameters passed to parent Agent class.

        Notes
        -----
        Specialized agent that orchestrates collaboration between multiple AI agents.
        Analyzes conversation context to determine which agent should handle each
        task and manages the overall workflow for complex optimization analysis.
        Initializes coordination counters and prompt templates.
        """
        super().__init__(
            name="Coordinator",
            description="This is a coordinator agent that chooses which agent to work on the problem next and organizes "
            "the conversation within its team. ",
            client=client,
            **kwargs,
        )

        self.agents: list[Agent] = agents
        self.max_rounds: int = max_rounds

        self.coordination_time: float = 0
        self.coordinator_success: bool = False
        self._init_cnt()
        self._init_prompt_template()

    def _init_cnt(self):
        """
        Initialize coordinator counters and status flags.

        Notes
        -----
        Sets up the retry counter and success flag for coordinator
        decision-making processes.
        """
        self.coordinator_cnt: int = 3
        self.coordinator_success: bool = False

    def _init_prompt_template(self):
        """
        Initialize prompt template and agent list for coordination.

        Notes
        -----
        Sets up the coordination prompt template and creates a formatted
        list of available agents with their descriptions.
        """
        self.prompt_template: str = get_prompts("coordinator_prompt")  # type: ignore
        self.agents_list: str = "".join(
            [
                "-" + agent.name + ": " + agent.description + "\n"
                for agent in self.agents
            ]
        )

    def generate_decision(
        self,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
        agent_name: Any,
        task: Any,
    ) -> Tuple[str, str | DecisionDict]:
        """
        Generate coordination decisions for multi-agent task assignment.

        Parameters
        ----------
        messages : list[ChatCompletionMessageParam]
            Original message history from the conversation.
        team_conversation : list[TeamConversationMessage]
            History of team conversation with agent responses.
        agent_name : object
            Streamlit text object to display the selected agent name.
        task : object
            Streamlit text object to display the assigned task.

        Returns
        -------
        Tuple[str, str | DecisionDict]
            Status of coordination ("In Progress", "Completed", "Terminated") and
            either the final output string or the decision dictionary.

        Notes
        -----
        Core coordination logic that analyzes conversation history and assigns
        tasks to appropriate agents. Includes retry logic and completion detection.
        """
        status = "In Progress"

        coordinate_prompt: str = self.prompt_template.format(agents=self.agents_list)
        pseudo_messages: list[ChatCompletionMessageParam] = (
            self.generate_pseudo_messages(
                messages, team_conversation, coordinate_prompt
            )
        )

        cnt: int = 3
        while cnt > 0:
            try:
                response: str = self.llm_call(  # type: ignore
                    messages=pseudo_messages, seed=cnt
                )
                decision: str = response.strip()
                if "```json" in decision:
                    decision = decision.split("```json")[1].split("```")[0]

                decision = decision.replace("\\", "")

                self.print_in_and_out(coordinate_prompt, response)
                logger.debug(f"Decision: {decision}")

                decision_dict: DecisionDict = json.loads(decision)

                if team_conversation:
                    # safeguard to prevent the coordinator from calling the agent
                    # after the user's query has been answered by explainer
                    if team_conversation[-1]["agent_name"] == "Explainer":
                        status = "Completed"
                        OptiChat_out = team_conversation[-1]["agent_response"]
                        if "DONE" in decision_dict.values():
                            logger.debug("DONE, the user's query is answered.")
                        else:
                            logger.debug(
                                "DONE, the user's query is answered, though the coordinator did not output 'DONE'."
                            )

                        return status, OptiChat_out

                    if (
                        "DONE" in decision_dict.values()
                        and team_conversation[-1]["agent_name"] == "Engineer"
                    ):
                        decision_dict: DecisionDict = {
                            "agent_name": "Explainer",
                            "task": "explain the technical feedback",
                        }

                else:
                    # the first round of the conversation
                    if "DONE" in decision_dict.values():
                        # sometimes user does not ask a question (e.g. saying 'thank you')
                        # and coordinator considers no query there and outputs 'DONE' directly
                        decision_dict: DecisionDict = {
                            "agent_name": "Explainer",
                            "task": "respond to the user",
                        }

                agent_name.text(decision_dict["agent_name"])
                task.text(decision_dict["task"])

                return status, decision_dict

            except Exception as e:
                logger.error(e)
                cnt -= 1
                logger.error("Invalid decision. Trying again ...")

                task.text(f"distribution failed ({cnt}/3)")

                if cnt == 0:
                    import traceback

                    err = traceback.format_exc()
                    logger.error(err)

                    status = "Terminated"
                    OptiChat_out = (
                        "LLM failed to assign tasks to experts! \n"
                        + "Error: "
                        + err
                        + "\n"
                    )

                    return status, OptiChat_out

        raise Exception("Unreachable code reached in generate_decision")

    def generate_decision_exp(
        self,
        args: Any,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
    ) -> DecisionDict | None:
        """
        Generate coordination decisions with experimental settings and simplified logic.

        Parameters
        ----------
        args : object
            Configuration arguments including temperature, json_mode settings.
        messages : list of dict
            Original message history from the conversation.
        team_conversation : list[TeamConversationMessage]
            History of team conversation with agent responses.

        Returns
        -------
        Dict[str, Any] | None
            Status of coordination and either the final output or decision dictionary.

        Notes
        -----
        Experimental version with simplified decision logic. If team conversation exists,
        automatically assigns Explainer; otherwise uses LLM for decision making.
        """
        self._init_cnt()
        coordinate_prompt: str = self.prompt_template.format(agents=self.agents_list)
        while self.coordinator_cnt > 0:
            # messages will only be updated outside the loop (in the OptiChat workflow fn)
            # team_conversation will be updated inside the loop (in the Engineer and Explainer fns)
            pseudo_messages = self.generate_pseudo_messages(
                messages, team_conversation, coordinate_prompt
            )
            try:
                # in current design, if coordinator has assigned the task once,
                # actually there will be no need to call llm to generate the decision again
                if team_conversation:
                    decision_dict: DecisionDict = {
                        "agent_name": "Explainer",
                        "task": "explain the technical feedback",
                    }
                    # last_agent = team_conversation[-1]["agent_name"]
                    # if last_agent == "Engineer":
                    #     decision = {'agent_name': 'Explainer', 'task': 'explain the technical feedback'}
                else:
                    response: str = self.llm_call_exp(
                        messages=pseudo_messages,
                        seed=self.coordinator_cnt,
                        temperature=args.temperature,
                        json_mode=args.json_mode,
                        stream=False,
                    )  # type: ignore
                    decision: str = response.strip()
                    if "```json" in decision:
                        decision = decision.split("```json")[1].split("```")[0]

                    decision = decision.replace("\\", "")

                    logger.debug(f"Decision: {decision}")

                    decision_dict: DecisionDict = json.loads(decision)
                    assert "agent_name" in decision_dict
                    assert decision_dict["agent_name"] in [
                        agent.name for agent in self.agents
                    ]
                    assert "task" in decision_dict

                    # in the first round of the conversation
                    # sometimes user does not ask a question (e.g. saying 'thank you')
                    # and coordinator considers no query there and outputs 'DONE' directly
                    if "DONE" in decision_dict.values():
                        decision_dict: DecisionDict = {
                            "agent_name": "Explainer",
                            "task": "respond to the user",
                        }

                self.coordinator_success = True
                return decision_dict

            except Exception as e:
                logger.error(e)
                self.coordinator_cnt -= 1
                logger.error("Invalid decision. Trying again ...")

                if self.coordinator_cnt == 0:
                    import traceback

                    err = traceback.format_exc()
                    logger.error(err)
                    return None


class Explainer(Agent):
    def __init__(self, client: Client, max_rounds: int = 5, **kwargs):
        """
        Initialize the Explainer agent for user-friendly explanations.

        Parameters
        ----------
        client : Client
            OpenAI client instance for making API calls.
        max_rounds : int, default=5
            Maximum number of explanation rounds before termination.
        **kwargs : dict
            Additional configuration parameters passed to parent Agent class.

        Notes
        -----
        Specialized agent that translates technical analysis results into
        user-friendly explanations suitable for non-technical stakeholders.
        Synthesizes feedback from Engineers and other technical agents into
        accessible natural language responses.
        """
        super().__init__(
            name="Explainer",
            description="This is an explainer agent whose task is to either (1) directly answer user queries if the questions can be analyzed through natural language only, or (2) summarize the technical feedback obtained from engineers to answer user queries",
            client=client,
            **kwargs,
        )
        self.explanation_time: float = 0
        self._init_prompt_template()

    def _init_prompt_template(self):
        """
        Initialize prompt template for explanation generation.

        Notes
        -----
        Sets up the explainer prompt template used to generate
        user-friendly explanations from technical feedback.
        """
        self.prompt_template: str = get_prompts("explainer_prompt")  # type: ignore

    def generate_explanation_exp(
        self,
        args: Any,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
    ) -> Stream[ChatCompletionChunk] | str:
        """
        Generate user-friendly explanations from technical analysis results.

        Parameters
        ----------
        args : object
            Configuration arguments including temperature and streaming settings.
        messages : list[ChatCompletionMessageParam]
            Original message history from the conversation.
        team_conversation : list[TeamConversationMessage]
            History of team conversation with technical feedback from other agents.

        Returns
        -------
        str
            Explanation response, either streamed or complete text.

        Notes
        -----
        Synthesizes technical feedback from Engineers and other agents into
        user-friendly explanations suitable for non-technical stakeholders.
        """
        prompt = self.prompt_template  # nothing to format here
        pseudo_messages = self.generate_pseudo_messages(
            messages, team_conversation, prompt
        )

        stream_or_completion: (
            Stream[ChatCompletionChunk] | str | None
        ) = self.llm_call_exp(
            messages=pseudo_messages,
            temperature=args.temperature,
            stream=args.explanation_stream,
        )  # type: ignore
        assert stream_or_completion is not None

        return stream_or_completion


class Engineer(Agent):
    """
    Specialized agent for performing technical analysis on optimization models.

    This agent handles complex optimization model analysis tasks including:
    - Syntax analysis and guidance generation
    - Tool calling for feasibility analysis, sensitivity analysis, etc.
    - Code generation and evaluation
    - Multi-step technical workflows

    The Engineer agent coordinates between multiple sub-capabilities to provide
    comprehensive technical feedback on optimization models.

    Attributes
    ----------
    syntax_cnt : int
        Counter for syntax analysis attempts
    operator_cnt : int
        Counter for operator/tool call attempts
    programmer_cnt : int
        Counter for code generation attempts
    evaluator_cnt : int
        Counter for code evaluation attempts
    various _success : bool
        Flags indicating success status for different operations
    various _time : float
        Timing measurements for different operation phases
    """

    def __init__(self, client: Client, **kwargs):
        """Initialize an Engineer agent specialized in technical optimization feedback.

        This constructor creates an agent focused on providing technical
        feedback for optimization-related queries, particularly when the
        user's question involves scenarios that differ from the current model.
        The Engineer agent executes tools and functions when direct interaction
        with optimization models is required.

        Parameters
        ----------
        client : Client
            The client interface for communication with the AI model.
        **kwargs
            Additional keyword arguments passed to the parent Agent class.
            See Agent.__init__ for supported parameters.

        Notes
        -----
        The Engineer agent is designed to provide technical rather than
        natural-language explanations, making it suitable for users who
        need detailed technical feedback about optimization scenarios.
        """
        super().__init__(
            name="Engineer",
            description="This is an engineer agent whose task is to execute tools and functions when user's query requires an interaction with optimization model. The engineer agent provides technical feedback instead of natural-language explanations."
            "Note that some ‘why’ questions are better answered with technical feedback."
            "These questions often involve scenarios that differ from the current model.",
            client=client,
            **kwargs,
        )

        self.pattern: str = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"

        self._init_prompt_template()

        self.syntax_time: float = 0
        self.programming_time: float = 0
        self.evaluation_time: float = 0

        self.programmer_cnt: int = 3
        self.evaluator_cnt: int = 3

        self.syntax_success: bool = False
        self.operator_success: bool = False
        self.programmer_success: bool = False
        self.evaluator_success: bool = False

        self.unparsed_queried_components = None
        self.queried_components = None
        self.queried_model = None
        self.queried_function = None

    def _init_prompt_template(self):
        """
        Initialize prompt templates for engineer tasks.

        Notes
        -----
        Sets up templates for syntax guidance, operator commands,
        code generation, evaluation, and testing prompts.
        """
        self.syntax_reminder_prompt_template: str = get_prompts(  # type: ignore
            "syntax_reminder_prompt"
        )
        self.operator_prompt_template: str = get_prompts(  # type: ignore
            "operator_prompt"
        )
        self.code_reminder_prompt_template: str = get_prompts(  # type: ignore
            "code_reminder_prompt"
        )
        self.programmer_prompt_template: str = get_prompts(  # type: ignore
            "programmer_prompt"
        )
        self.evaluator_prompt_template: str = get_prompts(  # type: ignore
            "evaluator_prompt"
        )
        self.test_prompt_template: str = get_prompts("test_prompt")  # type: ignore

    def _init_fake_team_conversation(
        self, team_conversation: List[TeamConversationMessage], code_wo_labels: str
    ) -> None:
        """
        Initialize fake team conversation with code context.

        Parameters
        ----------
        team_conversation : list[TeamConversationMessage]
            Current team conversation history.
        code_wo_labels : str
            Source code without labels for context.

        Notes
        -----
        Creates a deep copy of team conversation and adds code reminder
        to provide context for code generation tasks.
        """
        self.fake_team_conversation = copy.deepcopy(team_conversation)
        self.source_code = self.code_reminder_prompt_template.format(
            source_code=code_wo_labels
        )
        self.fake_team_conversation.append(
            {"agent_name": "Code reminder", "agent_response": self.source_code}
        )

    def _init_cnt(self) -> None:
        """
        Initialize all retry counters for engineer operations.

        Notes
        -----
        Resets all counter variables used for tracking retry attempts
        in syntax analysis, operator calls, programming, and evaluation phases.
        """
        self.syntax_cnt: int = 3
        self.operator_cnt: int = 3
        self.programmer_cnt: int = 3  # cnt for programmer output format
        self.evaluator_cnt: int = 3  # cnt for evaluator output format
        self.debug_times_left: int = (
            3  # cnt for debugging (format correct but not satisfactory code)
        )
        self.syntax_success: bool = False
        self.operator_success: bool = False
        self.programmer_success: bool = False
        self.evaluator_success: bool = False

        self.queried_components = None
        self.queried_model = None
        self.queried_function = None

    def execute_code(self, revision_code: str, print_code: str) -> Tuple[str, str]:
        """
        Execute generated code and return results.

        Parameters
        ----------
        revision_code : str
            Code modifications to append to the source code.
        print_code : str
            Code for printing/display purposes.

        Returns
        -------
        tuple of (str, str)
            Complete source code and execution results.

        Notes
        -----
        Combines source code with revision code, executes it, and saves
        both the complete code and execution results to log files for debugging.
        Updates fake team conversation with execution results.
        """
        # src_code = insert_code(self.source_code, revision_code, 'REVISION')
        # src_code = insert_code(src_code, print_code, 'PRINT')
        src_code: str = self.source_code + "\n" + revision_code
        execution_rst: str = run_with_exec(src_code)

        # save the complete code as .py
        with open(
            f"./logs/code_draft/complete_code_{self.debug_times_left}.py", "w"
        ) as f:
            f.write(src_code)
        # save the execution result as .txt
        with open(
            f"./logs/code_draft/execution_result_{self.debug_times_left}.txt", "w"
        ) as f:
            f.write(execution_rst)

        self.fake_team_conversation.append(
            {"agent_name": "Execution result", "agent_response": execution_rst}
        )
        return src_code, execution_rst

    def tool_call_exp(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        seed: int = 10,
        temperature: float = 0.1,
        is_syntax_guidance: bool = False,
        syntax_mode: str = "none",
    ):
        """
        Make experimental LLM calls with tool calling capabilities.

        Parameters
        ----------
        prompt : str, optional
            Single prompt string to send to the LLM.
        messages : list of dict, optional
            List of message dictionaries with 'role' and 'content' keys.
        seed : int, default=10
            Random seed for reproducible results.
        temperature : float, default=0.1
            Sampling temperature for response randomness.
        is_syntax_guidance : bool, default=False
            Whether to use syntax guidance tools.
        syntax_mode : str, default="none"
            Mode for syntax analysis ("single", "multiple", "none").

        Returns
        -------
        completion object
            LLM completion with tool calling capabilities enabled.

        Notes
        -----
        Extended LLM call interface with tool calling support for technical
        analysis tasks. Handles both syntax guidance and operational tools.
        """

        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)
        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if prompt is not None:
            messages = [
                ChatCompletionSystemMessageParam(
                    {"role": "system", "content": self.system_prompt}
                ),
                ChatCompletionUserMessageParam({"role": "user", "content": prompt}),
            ]

        if is_syntax_guidance:
            tools: list[ChatCompletionToolParam] = self.syntax_guidance_tool
            tool_choice: ChatCompletionToolChoiceOptionParam = {
                "type": "function",
                "function": {"name": "syntax_guidance"},
            }
        else:
            if syntax_mode == "multiple":
                tools = self.multiple_tools
            elif syntax_mode == "single":
                tools = self.single_tools
            elif syntax_mode == "none":
                tools = self.none_tools
            elif syntax_mode == "all":
                tools = self.all_tools
            else:
                raise Exception("Invalid mode!")

            tool_choice: ChatCompletionToolChoiceOptionParam = "required"

        if type(self.client) in [OpenAI, Client]:
            if self.llm not in ["o3"]:
                completion: ChatCompletion = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,  # type: ignore
                    seed=seed,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            else:
                completion: ChatCompletion = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,  # type: ignore
                    seed=seed,
                    tools=tools,
                    tool_choice=tool_choice,
                )

            if completion.choices[0].message.tool_calls:
                # internal tool is called
                fn_call = completion.choices[0].message.tool_calls[0].function
                fn_name = fn_call.name
                fn_args = fn_call.arguments
                logger.debug(f"function name = {fn_name}")
                logger.debug(f"function arguments = {fn_args}")
            else:
                raise Exception(
                    "No tool call executed by Operator, perhaps because of the 'auto' tool choice!"
                )
        else:
            raise Exception("Client type not supported!")

        return fn_name, fn_args

    def generate_syntax_exp(
        self,
        args: Any,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
        models_dict: ModelsContainer,
    ) -> Tuple[str, str]:
        """
        Generate syntax guidance for model analysis with experimental settings.

        Parameters
        ----------
        args : object
            Configuration arguments with experimental settings.
        messages : list[ChatCompletionMessageParam]
            Conversation message history.
        team_conversation : list[TeamConversationMessage]
            Team conversation history for context.
        models_dict : ModelsContainer
            Dictionary containing model representations.

        Returns
        -------
        tuple of (str, str) or (str, str)
            Syntax guidance output and syntax mode ("single", "multiple", "none").

        Notes
        -----
        Provides syntax reminders and guidance for technical analysis tools.
        Adapts function availability based on model type (LP vs IP).
        """
        while not self.syntax_success and self.syntax_cnt > 0:
            component_descriptions = extract_component_descriptions(models_dict)

            if models_dict["model_representation"]["model_type"] != "LP":
                function_names = [
                    fn for fn in self.function_names if fn != "sensitivity_analysis"
                ]
            else:
                function_names = [fn for fn in self.function_names]

            prompt = self.syntax_reminder_prompt_template.format(
                function_names=function_names,
                component_name_meaning_pairs=str(component_descriptions),
            )
            pseudo_messages = self.generate_pseudo_messages(
                messages, team_conversation, prompt
            )
            self.syntax_cnt -= 1
            try:
                syntax_start = time.time()
                fn_name, fn_args = self.tool_call_exp(
                    messages=pseudo_messages,
                    seed=self.syntax_cnt,
                    temperature=args.temperature,
                    is_syntax_guidance=True,
                )
                syntax_end = time.time()
                self.syntax_time += syntax_end - syntax_start

                self.queried_function = json.loads(fn_args).get("queried_function")
                self.queried_components = json.loads(fn_args).get("queried_components")
                self.queried_model = json.loads(fn_args).get("queried_model")
                # forced syntax_guidance to be called
                syntax_output, syntax_mode = syntax_guidance(
                    self.queried_function,
                    self.queried_components,
                    self.queried_model,
                    models_dict,
                )
                self.syntax_success = True
                return syntax_output, syntax_mode

            except Exception as e:
                logger.error(str(e))
                # import traceback
                # err = traceback.format_exc()
                # print(err)
                if self.syntax_cnt == 0:
                    self.syntax_success = False
                    return "LLM failed", "none"

        raise Exception("Should not reach here!")

    def generate_feedback_exp(
        self,
        args: Any,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
        models_dict: ModelsContainer,
        syntax_mode: str,
    ) -> str:
        """
        Generate technical feedback using tool calls and model analysis.

        Parameters
        ----------
        args : object
            Configuration arguments with experimental settings.
        messages : list[ChatCompletionMessageParam]
            Conversation message history.
        team_conversation : list[TeamConversationMessage]
            Team conversation history for context.
        models_dict : ModelsContainer
            Dictionary containing model representations.
        syntax_mode : str
            Mode for syntax analysis ("single", "multiple", "none").

        Returns
        -------
        str
            Technical feedback from tool execution and analysis.

        Notes
        -----
        Orchestrates tool calling to perform technical analysis on optimization
        models. Includes error handling and retry logic for robustness.
        """
        while not self.operator_success and self.operator_cnt > 0:
            prompt = self.operator_prompt_template  # nothing to format here
            pseudo_messages: list[ChatCompletionMessageParam] = (
                self.generate_pseudo_messages(messages, team_conversation, prompt)
            )
            self.operator_cnt -= 1
            try:
                syntax_start: float = time.time()
                fn_name: str
                fn_args: str
                fn_name, fn_args = self.tool_call_exp(
                    messages=pseudo_messages,
                    seed=self.operator_cnt,
                    temperature=args.temperature,
                    is_syntax_guidance=False,
                    syntax_mode=syntax_mode,
                )
                syntax_end: float = time.time()
                self.syntax_time += syntax_end - syntax_start

                self.queried_function = fn_name
                self.queried_model = json.loads(fn_args).get("queried_model")
                self.unparsed_queried_components = json.loads(fn_args).get(
                    "queried_components"
                )
                self.queried_components = fnArgsDecoder(
                    self.unparsed_queried_components
                )

                # pass the function name and arguments to the function
                if fn_name == "feasibility_restoration":
                    fn_output = feasibility_restoration(
                        self.queried_components, self.queried_model, models_dict
                    )
                elif fn_name == "sensitivity_analysis":
                    fn_output = sensitivity_analysis(
                        self.queried_components, self.queried_model, models_dict
                    )
                elif fn_name == "components_retrival":
                    fn_output = components_retrival(
                        self.queried_components, self.queried_model, models_dict
                    )
                elif fn_name == "evaluate_modification":
                    fn_output = evaluate_modification(
                        self.queried_components, self.queried_model, models_dict
                    )
                else:
                    raise Exception("invalid function name")

                self.operator_success = True
                return fn_output

            except Exception as e:
                logger.error(e)
                import traceback

                err = traceback.format_exc()
                # embed the error message into the syntax reminder in team_conversation
                error_response: str = (
                    "\n\nProblematic queried_components: "
                    + f"{self.unparsed_queried_components} \n\nError: {err}"
                )
                team_conversation.append(
                    {"agent_name": "Execution result", "agent_response": error_response}
                )

                if self.operator_cnt == 0:
                    self.operator_success = False
                    return "LLM failed"

        raise Exception("Should not reach here!")

    def programmer_loop_exp(
        self, args: Any, pseudo_messages: List[ChatCompletionMessageParam]
    ) -> Tuple[str, str, str] | Tuple[None, None, None]:
        """
        Loop to generate code solutions with retry logic.

        Parameters
        ----------
        args : object
            Configuration arguments with experimental settings.
        pseudo_messages : list[ChatCompletionMessageParam]
            Message history including problem context.

        Returns
        -------
        tuple of (str, str, str) or (None, None, None)
            Code output, revision code, and print code.
            Returns None tuple if all retries failed.

        Notes
        -----
        Generates code solutions for optimization problems with retry logic.
        Extracts revision and print code from LLM output for execution.
        """
        while self.programmer_cnt > 0:
            program_start = time.time()
            code_output: str = self.llm_call_exp(
                messages=pseudo_messages,
                seed=self.programmer_cnt,
                temperature=args.temperature,
                stream=False,
            )  # type: ignore
            program_end = time.time()
            self.programming_time += program_end - program_start

            self.programmer_cnt -= 1
            try:
                snippets = re.findall(self.pattern, code_output, flags=re.DOTALL)
                # assert len(snippets) <= 2
                # assert snippets[0][0] == 'python'
                # assert snippets[1][0] == 'python'
                # revision_code = snippets[0][1]
                # print_code = snippets[1][1]
                revision_code = snippets[0][1]
                print_code = ""
                self.programmer_success = True
                self.fake_team_conversation.append(
                    {"agent_name": "Programmer", "agent_response": code_output}
                )
                return code_output, revision_code, print_code

            except AssertionError as e:
                logger.error(e)
                # import traceback
                # err = traceback.format_exc()
                # print(err)
                if self.programmer_cnt == 0:
                    self.programmer_success = False
                    return None, None, None

        raise Exception("Should not reach here!")

    def evaluator_loop_exp(
        self, args: Any, pseudo_messages: List[ChatCompletionMessageParam]
    ) -> Tuple[str, str, str] | Tuple[None, None, None]:
        """
        Loop to evaluate generated code with retry logic.

        Parameters
        ----------
        args : object
            Configuration arguments with experimental settings.
        pseudo_messages : list[ChatCompletioinnMessageParam]
            Message history including code context.

        Returns
        -------
        tuple of (str, str, str) or (None, None, None)
            Evaluation output, decision (approve/reject), and comments.
            Returns None tuple if all retries failed.

        Notes
        -----
        Evaluates generated code for correctness and provides feedback.
        Includes retry logic and error handling for robust evaluation.
        """
        while self.evaluator_cnt > 0:
            evaluation_start: float = time.time()
            # evaluation_output = self.llm_call_exp(messages=pseudo_messages,
            #                                       seed=self.evaluator_cnt, temperature=args.temperature,
            #                                       json_mode=args.json_mode, stream=False)
            evaluation_output: str = self.llm_call_exp(
                messages=pseudo_messages,
                seed=self.evaluator_cnt,
                temperature=args.temperature,
                json_mode=True,
                stream=False,
            )  # type: ignore
            evaluation_end: float = time.time()
            self.evaluation_time += evaluation_end - evaluation_start

            self.evaluator_cnt -= 1
            try:
                # evaluation = evaluation_output.strip()
                # if "```json" in evaluation:
                #     evaluation = evaluation.split("```json")[1].split("```")[0]
                # evaluation = evaluation.replace("\\", "")
                # # print('Code review:', evaluation)
                # evaluation = json.loads(evaluation)

                # delete until the first '```json'
                if "```json" in evaluation_output:
                    evaluation_output = evaluation_output[
                        evaluation_output.find("```json") + 7 :
                    ]
                    evaluation_output = evaluation_output[
                        : evaluation_output.rfind("```")
                    ]

                start: int = evaluation_output.find("{")
                end: int = evaluation_output.rfind("}")
                evaluation_output = evaluation_output[start : end + 1]
                evaluation: dict[str, Any] = json.loads(evaluation_output)
                decision: str = evaluation["decision"]
                comment: str = evaluation["comment"]

                self.evaluator_success = True
                self.fake_team_conversation.append(
                    {"agent_name": "Evaluator", "agent_response": evaluation_output}
                )
                return evaluation_output, decision, comment

            except AssertionError as e:
                logger.error(e)
                # import traceback
                # err = traceback.format_exc()
                # print(err)
                if self.evaluator_cnt == 0:
                    self.evaluator_success = False
                    return None, None, None

        raise Exception("Should not reach here!")

    def generate_code_exp(
        self,
        args: Any,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
        models_dict: ModelsContainer,
    ) -> Tuple[str, str, str]:
        """
        Generate and evaluate code solutions through iterative development.

        Parameters
        ----------
        args : object
            Configuration arguments with experimental settings.
        messages : list[ChatCompletionMessageParam]
            Conversation message history.
        team_conversation : list[TeamConversationMessage]
            Team conversation history for context.
        models_dict : ModelsContainer
            Dictionary containing model representations.

        Returns
        -------
        tuple of (str, str, str) or (None, None, None)
            Code output, execution results, and evaluation output.
            Returns None tuple if all retries failed.

        Notes
        -----
        Orchestrates iterative code generation, execution, and evaluation.
        Includes debugging loops with programmer and evaluator agents working
        together to create working code solutions.
        """
        # initialize
        self._init_prompt_template()
        self._init_fake_team_conversation(
            team_conversation, models_dict["model_representation"]["code"]
        )

        # until the programmer generates the code that evaluator approves
        while self.debug_times_left > 0:
            # Only init the cnt for programmer and evaluator for every debugging loop
            # because _init_cnt() will reset all the success, cnt, debug_times_left
            self.programmer_cnt = 2
            self.evaluator_cnt = 2

            # until the programmer generates the code in correct format
            programmer_prompt = self.programmer_prompt_template
            pseudo_messages = self.generate_pseudo_messages(
                messages, self.fake_team_conversation, programmer_prompt
            )
            code_output, revision_code, print_code = self.programmer_loop_exp(
                args, pseudo_messages
            )
            if (
                not self.programmer_success
                or code_output is None
                or revision_code is None
                or print_code is None
            ):
                return "LLM failed", "None", "None"

            # simply executing the code
            complete_code, execution_rst = self.execute_code(revision_code, print_code)

            # until the evaluator evaluates the code in correct format
            evaluator_prompt = self.evaluator_prompt_template
            pseudo_messages = self.generate_pseudo_messages(
                messages, self.fake_team_conversation, evaluator_prompt
            )
            evaluation_output, decision, comment = self.evaluator_loop_exp(
                args, pseudo_messages
            )
            if (
                not self.evaluator_success
                or evaluation_output is None
                or decision is None
                or comment is None
            ):
                return code_output, execution_rst, "LLM failed"

            if decision == "accept":
                return code_output, execution_rst, evaluation_output
            else:
                self.debug_times_left -= 1
                if self.debug_times_left == 0:
                    # return the last evaluation output though it is rejected by evaluator
                    return code_output, execution_rst, evaluation_output

        raise Exception("Should not reach here!")

    def generate_report_exp(
        self,
        args: Any,
        messages: List[ChatCompletionMessageParam],
        team_conversation: List[TeamConversationMessage],
        models_dict: ModelsContainer,
    ) -> Tuple[List[ChatCompletionMessageParam], List[TeamConversationMessage]]:
        """
        Generate comprehensive technical report with experimental settings.

        Parameters
        ----------
        args : object
            Configuration arguments with experimental settings.
        messages : list[ChatCompletionMessageParam]
            Conversation message history.
        team_conversation : list[TeamConversationMessage]
            Team conversation history for context.
        models_dict : ModelsContainer
            Dictionary containing model representations.

        Returns
        -------
        list of dict
            Updated team conversation with technical analysis results.

        Notes
        -----
        Orchestrates complete technical analysis workflow including syntax
        guidance, tool calling, and code generation. Handles both internal
        and external experiment modes.
        """
        self._init_cnt()

        if args.external_experiment:
            syntax_output, syntax_mode = "external_tools", "none"
            self.syntax_success = True
        else:
            syntax_output, syntax_mode = self.generate_syntax_exp(
                args, messages, team_conversation, models_dict
            )

        if not self.syntax_success:
            team_conversation.append(
                {"agent_name": "Syntax reminder", "agent_response": syntax_output}
            )
            messages.append(
                ChatCompletionAssistantMessageParam(
                    {"role": "assistant", "content": syntax_output}
                )
            )
        else:
            if syntax_output != "external_tools":
                # add code reminder to the team_conversation as well to help find correct component indexes
                # add syntax reminder
                team_conversation.append(
                    {
                        "agent_name": "Code reminder",
                        "agent_response": models_dict["model_representation"]["code"],
                    }
                )
                team_conversation.append(
                    {"agent_name": "Syntax reminder", "agent_response": syntax_output}
                )
                function_output = self.generate_feedback_exp(
                    args, messages, team_conversation, models_dict, syntax_mode
                )

                team_conversation = [
                    item
                    for item in team_conversation
                    if item["agent_name"] not in ["Code reminder", "Syntax reminder"]
                ]

                team_conversation.append(
                    {"agent_name": "Operator", "agent_response": function_output}
                )
                if self.operator_success:
                    messages.append(
                        ChatCompletionAssistantMessageParam(
                            {"role": "assistant", "content": function_output}
                        )
                    )
                else:
                    syntax_output = "external_tools"

            if not args.internal_experiment:
                if syntax_output == "external_tools":
                    code_output, execution_rst, evaluation_output = (
                        self.generate_code_exp(
                            args, messages, team_conversation, models_dict
                        )
                    )
                    team_conversation.append(
                        {"agent_name": "Programmer", "agent_response": code_output}
                    )
                    team_conversation.append(
                        {
                            "agent_name": "Execution result",
                            "agent_response": execution_rst,
                        }
                    )
                    team_conversation.append(
                        {"agent_name": "Evaluator", "agent_response": evaluation_output}
                    )

                    messages.append(
                        ChatCompletionAssistantMessageParam(
                            {
                                "role": "assistant",
                                "content": "Programmer:\n\n" + code_output,
                            }
                        )
                    )
                    messages.append(
                        ChatCompletionAssistantMessageParam(
                            {
                                "role": "assistant",
                                "content": "Execution result:\n\n" + execution_rst,
                            }
                        )
                    )
                    messages.append(
                        ChatCompletionAssistantMessageParam(
                            {
                                "role": "assistant",
                                "content": "Evaluator:\n\n" + evaluation_output,
                            }
                        )
                    )

        return messages, team_conversation

    def generate_test_result_exp(
        self, args: Any, messages: List[ChatCompletionMessageParam], gt_a: str
    ) -> str:
        """
        Generate test results by comparing with ground truth answer.

        Parameters
        ----------
        args : object
            Configuration arguments including temperature settings.
        messages : list[ChatCompletionMessageParam]
            Conversation message history.
        gt_a : str
            Ground truth answer from human expert.

        Returns
        -------
        str
            Test evaluation result (pass/fail assessment).

        Notes
        -----
        Uses test prompt template to evaluate model responses against
        human expert answers for validation purposes.
        """
        self._init_prompt_template()
        prompt = self.test_prompt_template.format(human_expert_answer=gt_a)
        pseudo_messages = self.generate_pseudo_messages(messages, [], prompt)
        pass_or_fail: str = self.llm_call_exp(
            messages=pseudo_messages, temperature=args.temperature, stream=False
        )  # type: ignore
        return pass_or_fail
