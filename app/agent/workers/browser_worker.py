"""
Browser Worker - Specialized agent for browser operations.
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.terminate import Terminate


class BrowserWorker(ToolCallAgent):
    """
    Specialized worker for browser operations.
    
    Handles tasks like:
    - Web page navigation
    - Form filling
    - Page interaction
    - Content extraction from web pages
    """

    name: str = "BrowserWorker"
    description: str = (
        "专注于浏览器操作的 Agent，负责网页浏览、表单填写、页面交互等任务"
    )

    system_prompt: str = """你是一个浏览器操作专家 Agent。你的职责是：
1. 浏览和导航网页
2. 与网页元素进行交互（点击、输入、滚动等）
3. 从网页中提取所需信息
4. 填写表单和提交数据

请确保：
- 在操作前等待页面完全加载
- 处理可能出现的弹窗和确认框
- 提取的信息准确完整
- 遇到问题时提供清晰的错误说明

**重要**：当你完成浏览器操作任务并获取到所需信息后，必须调用 `terminate` 工具结束任务，并在最后一次回复中总结获取的信息。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    max_steps: int = 0  # Unlimited steps
