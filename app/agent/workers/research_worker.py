"""
Research Worker - Specialized agent for comprehensive research.
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.terminate import Terminate
from app.tool.web_search import WebSearch


class ResearchWorker(ToolCallAgent):
    """
    Specialized worker for comprehensive research.
    
    Combines search and browsing capabilities for in-depth research.
    Handles tasks like:
    - Deep research on topics
    - Fact verification
    - Comparative analysis
    - Trend analysis
    """

    name: str = "ResearchWorker"
    description: str = (
        "综合研究 Agent，结合搜索和分析能力进行深度研究"
    )

    system_prompt: str = """你是一个综合研究 Agent。你的职责是：
1. 使用搜索工具查找相关信息
2. 浏览网页获取详细内容
3. 分析和整理搜集到的信息
4. 提供有深度的研究结论

研究流程：
1. 明确研究问题和目标
2. 制定搜索策略
3. 收集多方来源的信息
4. 验证信息的准确性
5. 综合分析得出结论

输出要求：
- 提供信息来源
- 标注信息的时效性
- 指出不确定或有争议的内容
- 给出基于证据的结论

**重要**：当你完成研究任务并得到足够的结论后，必须调用 `terminate` 工具结束任务，并在最后一次回复中总结研究结果。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(WebSearch(), BrowserUseTool(), Terminate())
    )

    max_steps: int = 0  # Unlimited steps
