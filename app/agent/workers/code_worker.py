"""
Code Worker - Specialized agent for code execution.
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.terminate import Terminate


class CodeWorker(ToolCallAgent):
    """
    Specialized worker for code execution tasks.
    
    Handles tasks like:
    - Writing and running Python code
    - Data processing and transformation
    - Automation scripts
    - Calculations and computations
    """

    name: str = "CodeWorker"
    description: str = (
        "专注于代码执行的 Agent，负责编写和运行 Python 代码、数据处理、自动化脚本等"
    )

    system_prompt: str = """你是一个代码执行专家 Agent。你的职责是：
1. 编写清晰、高效的 Python 代码
2. 执行数据处理和转换任务
3. 实现自动化脚本
4. 进行数学计算和数据分析

请确保：
- 代码有适当的注释和文档
- 处理可能的异常情况
- 输出结果清晰易懂
- 使用标准库和常用第三方库

**重要**：当你完成代码执行任务并得到结果后，必须调用 `terminate` 工具结束任务，并在最后一次回复中总结执行结果。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PythonExecute(), Terminate())
    )

    max_steps: int = 0  # Unlimited steps
