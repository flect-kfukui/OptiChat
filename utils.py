import json
import time
from typing import Any, List, NamedTuple

# Streamlit
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from loguru import logger

# GPT
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from agents import Coordinator, Engineer, Explainer, Interpreter

# Import the TeamConversationMessage type
# OptiChat types
from optichat_types import (
    DecisionDict,
    ModelsContainer,
    TeamConversationMessage,
    WorkflowResult,
)
from prompts import get_syntax_guidance_tool, get_tools

_ = load_dotenv(find_dotenv())  # read local .env file


class AgentsContainer(NamedTuple):
    """Container for all OptiChat agent instances.

    Groups together all the main agent types used in the OptiChat system
    for multi-agent optimization analysis.
    """

    interpreter: Interpreter
    explainer: Explainer
    engineer: Engineer
    coordinator: Coordinator


def get_agents(
    fn_names: List[str], client: OpenAI, llm: str = "gpt-4-turbo-preview"
) -> AgentsContainer:
    """
    Initialize and configure all agent instances for the OptiChat system.

    Parameters
    ----------
    fn_names : list of str
        List of function names available for the engineer agent's tools.
    client : OpenAI.Client
        OpenAI client instance for API calls.
    llm : str, default="gpt-4-turbo-preview"
        Language model identifier to use for all agents.

    Returns
    -------
    AgentsContainer
        Configured agent instances ready for optimization model analysis:
        - Interpreter: Analyzes and describes model components
        - Explainer: Provides detailed explanations of results
        - Engineer: Performs technical analysis and tool calls
        - Coordinator: Orchestrates multi-agent collaboration

    Notes
    -----
    Configures engineer with appropriate tools based on available functions
    and sets up coordinator with explainer and engineer as sub-agents.
    """
    interpreter = Interpreter(client=client, llm=llm)
    explainer = Explainer(client=client, llm=llm)

    multiple_tools, single_tools, none_tools, all_tools, tool_choice = get_tools(
        fn_names
    )
    syntax_guidance_tool = get_syntax_guidance_tool()
    engineer = Engineer(
        client=client,
        llm=llm,
        multiple_tools=multiple_tools,
        single_tools=single_tools,
        none_tools=none_tools,
        all_tools=all_tools,
        tool_choice=tool_choice,
        syntax_guidance_tool=syntax_guidance_tool,
        function_names=fn_names,
    )
    coordinator = Coordinator(client=client, agents=[explainer, engineer], llm=llm)
    return AgentsContainer(
        interpreter=interpreter,
        explainer=explainer,
        engineer=engineer,
        coordinator=coordinator,
    )


def save_team_conversation(
    team_conversation: List[TeamConversationMessage], filename: str
) -> None:
    """
    Save team conversation history to a file.

    Parameters
    ----------
    team_conversation : list[TeamConversationMessage]
        List of conversation messages from different agents.
    filename : str
        Path to the output file for saving the conversation.

    Notes
    -----
    Saves each conversation message as a separate JSON line in the file
    for later analysis and debugging purposes.
    """
    with open(filename, "w") as f:
        for message in team_conversation:
            f.write(json.dumps(message) + "\n")


def OptiChat_workflow_exp(
    args: Any,
    coordinator: Coordinator,
    engineer: Engineer,
    explainer: Explainer,
    messages: List[ChatCompletionMessageParam],
    models_dict: ModelsContainer,
) -> WorkflowResult:
    """
    Execute the main OptiChat workflow with multi-agent coordination.

    Parameters
    ----------
    args : object
        Configuration object containing experiment settings and parameters.
    coordinator : Coordinator
        Coordinator agent instance for orchestrating multi-agent collaboration.
    engineer : Engineer
        Engineer agent instance for technical analysis and tool execution.
    explainer : Explainer
        Explainer agent instance for providing detailed explanations.
    messages : list of dict
        Chat message history with role and content keys.
    models_dict : ModelsContainer
        Dictionary containing optimization model representations and metadata.

    Returns
    -------
    WorkflowResult
        - Updated messages list with new assistant responses
        - Team conversation history with agent interactions

    Notes
    -----
    Implements a multi-round conversation system where:
    1. Coordinator decides which agent should handle the current query
    2. Selected agent (Engineer or Explainer) processes the request
    3. Results are integrated back into conversation history
    4. Process continues until completion or max rounds reached

    Tracks timing for different agent operations for performance analysis.
    """
    team_conversation: List[TeamConversationMessage] = []
    rounds: int = 0

    # set the time in agents to 0
    coordinator.coordination_time = 0
    engineer.syntax_time = 0
    engineer.programming_time = 0
    engineer.evaluation_time = 0
    explainer.explanation_time = 0

    # in current design, if coordinator has assigned the task once,
    # actually there will be no need to call llm to generate the decision again
    while rounds <= coordinator.max_rounds:
        coordinator_start: float = time.time()
        decision: DecisionDict | None = coordinator.generate_decision_exp(
            args, messages, team_conversation
        )
        coordinator_end: float = time.time()
        coordinator.coordination_time += coordinator_end - coordinator_start

        if not decision:
            logger.error("coordinator failed to generate decision")
            messages.append(
                ChatCompletionAssistantMessageParam(
                    {"role": "assistant", "content": "LLM failed"}
                )
            )
            return WorkflowResult(
                messages=messages, team_conversation=team_conversation
            )

        else:
            if decision["agent_name"] == "Engineer":
                # unlike explainer, engineer team has already updated the
                # team_conversation and messages in fn below
                # syntax time, programming time, evaluation time are also
                # updated in the fn below
                messages, team_conversation = engineer.generate_report_exp(
                    args, messages, team_conversation, models_dict
                )

            elif decision["agent_name"] == "Explainer":
                explainer_start: float = time.time()
                explanation: Stream[ChatCompletionChunk] | str = (
                    explainer.generate_explanation_exp(
                        args, messages, team_conversation
                    )
                )
                if args.explanation_stream and isinstance(explanation, Stream):
                    with st.chat_message("assistant"):
                        explanation_response: Any = st.write_stream(explanation)
                else:
                    explanation_response = explanation

                explainer_end: float = time.time()
                explainer.explanation_time += explainer_end - explainer_start

                team_conversation.append(
                    {"agent_name": "Explainer", "agent_response": explanation_response}
                )
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        {"role": "assistant", "content": explanation_response}
                    )
                )
                return WorkflowResult(
                    messages=messages, team_conversation=team_conversation
                )

            else:
                raise ValueError(
                    f"Decision {decision} has an invalid agent name. "
                    f"Please choose from Engineer or Explainer."
                )

        rounds += 1

    # Return messages and team_conversation if max rounds reached
    return WorkflowResult(messages=messages, team_conversation=team_conversation)
