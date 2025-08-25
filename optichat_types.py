"""Type definitions for the OptiChat system."""

from typing import TypedDict


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
