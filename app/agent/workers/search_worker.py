"""
Search Worker - Specialized agent for web searching.
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.terminate import Terminate
from app.tool.web_search import WebSearch


class SearchWorker(ToolCallAgent):
    """
    Specialized worker for web searching.
    
    Handles tasks like:
    - Information retrieval
    - Data research
    - Real-time data fetching
    - Fact checking
    """

    name: str = "SearchWorker"
    description: str = (
        "专注于网络搜索的 Agent，负责信息检索、资料查询、实时数据获取等"
    )

    system_prompt: str = """你是一个网络搜索专家 Agent。你的职责是：
1. 使用搜索引擎查找相关信息
2. 筛选和验证搜索结果
3. 提取关键信息和数据
4. 提供信息来源引用

请确保：
- 使用精确的搜索关键词
- 验证信息的准确性
- 提供多个来源的信息
- 标注信息的时效性

**重要**：当你完成搜索任务并收集到足够的信息后，必须调用 `terminate` 工具结束任务，并在最后一次回复中总结你找到的所有信息。不要无休止地搜索，找到关键信息后就应该结束。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(WebSearch(), Terminate())
    )

    max_steps: int = 0  # Unlimited steps
