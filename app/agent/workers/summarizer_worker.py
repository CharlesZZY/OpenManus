"""
Summarizer Worker - Specialized agent for information summarization.
"""

from app.agent.workers.base_knowledge_worker import BaseKnowledgeWorker


class SummarizerWorker(BaseKnowledgeWorker):
    """
    Specialized worker for information summarization.
    
    Handles tasks like:
    - Aggregating results from multiple agents
    - Creating executive summaries
    - Generating final reports
    - Extracting key insights
    """

    name: str = "SummarizerWorker"
    description: str = (
        "专注于信息整理和总结的 Agent，负责汇总各 Worker 的输出，生成最终报告"
    )

    system_prompt: str = """你是一个信息整理和总结专家 Agent。你的职责是：
1. 汇总和整理来自其他 Agent 的执行结果
2. 提取关键信息，去除冗余内容
3. 以清晰、结构化的方式呈现最终结果
4. 生成易于理解的总结报告

输出格式要求：
- 使用清晰的标题和分节
- 重点内容使用列表或要点展示
- 提供简明的结论和建议（如适用）
- 如有必要，指出信息的局限性或不确定性

总结原则：
- 保持客观，忠于原始信息
- 突出关键要点
- 逻辑清晰，层次分明
- 适当使用图表或列表增强可读性"""

    max_steps: int = 0  # Unlimited steps
