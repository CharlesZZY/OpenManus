"""
Base Knowledge Worker - Base class for knowledge-based workers that primarily use LLM capabilities.
"""

from typing import Optional

from pydantic import Field

from app.agent.react import ReActAgent
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice


class BaseKnowledgeWorker(ReActAgent):
    """
    Base class for knowledge-based workers.
    
    These workers primarily rely on LLM capabilities for reasoning and generation,
    without needing external tools. They're suitable for tasks like:
    - Mathematical reasoning
    - Content writing
    - Historical analysis
    - Information summarization
    
    Knowledge workers automatically terminate after generating their response,
    as they typically complete their task in a single step.
    """

    name: str = "KnowledgeWorker"
    description: str = "A knowledge-based worker that uses LLM capabilities"

    # Knowledge workers don't use tools by default
    tool_choices: str = ToolChoice.NONE

    # Lower max_steps since knowledge workers typically complete in fewer steps
    max_steps: int = 0  # Unlimited steps

    async def think(self) -> bool:
        """
        Use LLM for reasoning without tool calls.
        
        Returns:
            True if the LLM produced a response, False otherwise
        """
        # Build the prompt
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # Call LLM without tools
        response = await self.llm.ask(
            messages=self.messages,
            system_msgs=(
                [Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None
            ),
        )

        if response:
            self.memory.add_message(Message.assistant_message(response))
            return True

        return False

    async def act(self) -> str:
        """
        Return the LLM's response as the action result.
        
        For knowledge workers, the "action" is simply returning
        the generated content and then terminating.
        
        Returns:
            The last assistant message content
        """
        result = "No response generated"
        if self.messages and self.messages[-1].content:
            result = self.messages[-1].content
        
        # Knowledge workers automatically terminate after generating their response
        # since they typically complete their task in a single step
        logger.info(f"📝 {self.name} completed task, auto-terminating...")
        self.state = AgentState.FINISHED
        
        return result
