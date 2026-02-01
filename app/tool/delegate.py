"""
Delegate Tool - Tool for Coordinator to delegate tasks to workers.
"""

from typing import Optional

from app.tool.base import BaseTool, ToolResult


class DelegateTask(BaseTool):
    """
    Tool for delegating tasks to worker agents.
    
    This tool is used by the Coordinator to assign tasks to specialized workers.
    The actual delegation is handled by the Coordinator agent, this tool
    provides the interface for the LLM to specify delegation intentions.
    """

    name: str = "delegate_task"
    description: str = """将任务委派给指定的 Worker Agent 执行。

可用的 Workers:
- browser: 浏览器操作（网页浏览、表单填写、页面交互）
- code: 代码执行（Python 代码编写和运行）
- file: 文件操作（文件创建、编辑、读取）
- search: 网络搜索（信息检索、资料查询）
- math: 数学问题（数学推理、公式推导、计算）
- copywriter: 文案创作（文章、营销内容、邮件）
- history: 历史问题（历史事件、人物、时代背景）
- summarizer: 信息总结（汇总结果、生成报告）
- research: 综合研究（深度研究、多来源验证）
- data_analyst: 数据分析（数据处理、统计分析、可视化）

使用此工具时，请指定要委派的 worker 和具体任务描述。"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "worker": {
                "type": "string",
                "description": "要委派任务的 Worker 类型",
                "enum": [
                    "browser",
                    "code",
                    "file",
                    "search",
                    "math",
                    "copywriter",
                    "history",
                    "summarizer",
                    "research",
                    "data_analyst",
                ],
            },
            "task": {
                "type": "string",
                "description": "要委派的具体任务描述，需要清晰明确",
            },
            "context": {
                "type": "string",
                "description": "可选的上下文信息，帮助 Worker 更好地理解任务",
            },
        },
        "required": ["worker", "task"],
    }

    async def execute(
        self,
        worker: str,
        task: str,
        context: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute the delegation.
        
        Note: The actual execution is handled by the Coordinator.
        This method returns information about the delegation request.
        
        Args:
            worker: The worker type to delegate to
            task: The task description
            context: Optional context information
        
        Returns:
            ToolResult with delegation information
        """
        # This tool is a marker - actual delegation is handled by Coordinator
        delegation_info = {
            "worker": worker,
            "task": task,
            "context": context,
            "status": "pending_delegation",
        }

        return ToolResult(
            output=f"任务已准备委派给 {worker}:\n任务: {task}"
            + (f"\n上下文: {context}" if context else "")
        )
