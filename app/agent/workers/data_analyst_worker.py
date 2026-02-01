"""
Data Analyst Worker - Specialized agent for data analysis.
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate


class DataAnalystWorker(ToolCallAgent):
    """
    Specialized worker for data analysis.
    
    Combines code execution and file operations for data tasks.
    Handles tasks like:
    - Data cleaning and preprocessing
    - Statistical analysis
    - Data visualization
    - Report generation
    """

    name: str = "DataAnalystWorker"
    description: str = (
        "数据分析 Agent，负责数据处理、统计分析、可视化"
    )

    system_prompt: str = """你是一个数据分析专家 Agent。你的职责是：
1. 数据清洗和预处理
2. 统计分析和数据挖掘
3. 生成数据可视化图表
4. 提供数据驱动的洞察和建议

分析流程：
1. 理解数据和分析目标
2. 数据探索和质量检查
3. 数据清洗和转换
4. 执行统计分析
5. 生成可视化
6. 解读结果并给出建议

技术栈：
- pandas: 数据处理
- numpy: 数值计算
- matplotlib/seaborn: 可视化
- scipy/statsmodels: 统计分析

输出要求：
- 提供清晰的分析方法说明
- 解释统计指标的含义
- 图表要有适当的标题和标签
- 给出可操作的建议

**重要**：当你完成数据分析任务后，必须调用 `terminate` 工具结束任务，并在最后一次回复中总结分析结果。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PythonExecute(), StrReplaceEditor(), Terminate())
    )

    max_steps: int = 0  # Unlimited steps
