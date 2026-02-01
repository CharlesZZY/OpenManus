"""
File Worker - Specialized agent for file operations.
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate


class FileWorker(ToolCallAgent):
    """
    Specialized worker for file operations.
    
    Handles tasks like:
    - Creating new files
    - Editing existing files
    - Reading file contents
    - Organizing files and directories
    """

    name: str = "FileWorker"
    description: str = (
        "专注于文件操作的 Agent，负责文件创建、编辑、读取、组织等任务"
    )

    system_prompt: str = """你是一个文件操作专家 Agent。你的职责是：
1. 创建和编辑各种类型的文件
2. 读取和分析文件内容
3. 组织和管理文件结构
4. 进行文本替换和格式化

请确保：
- 在修改前备份重要文件
- 使用正确的文件编码
- 保持文件格式一致性
- 提供清晰的操作日志

**重要**：当你完成文件操作任务后，必须调用 `terminate` 工具结束任务，并在最后一次回复中总结操作结果。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(StrReplaceEditor(), Terminate())
    )

    max_steps: int = 0  # Unlimited steps
