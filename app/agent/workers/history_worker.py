"""
History Worker - Specialized agent for historical questions.
"""

from app.agent.workers.base_knowledge_worker import BaseKnowledgeWorker


class HistoryWorker(BaseKnowledgeWorker):
    """
    Specialized worker for historical question answering.
    
    Handles tasks like:
    - Historical event analysis
    - Historical figure introductions
    - Era and period explanations
    - Historical context and background
    """

    name: str = "HistoryWorker"
    description: str = (
        "专注于历史问题解答的 Agent，擅长历史事件分析、人物介绍、时代背景解读"
    )

    system_prompt: str = """你是一个历史学专家 Agent。你的职责是：
1. 准确回答历史相关问题
2. 提供历史事件的背景、原因、影响分析
3. 介绍历史人物的生平和贡献
4. 解释历史概念和时代特征
5. 必要时指出历史事件之间的联系和影响

请确保信息准确，并注明不确定的内容。

回答结构：
- 直接回答问题
- 提供历史背景
- 分析原因和影响
- 相关事件或人物的关联
- 历史意义或启示（如适用）"""

    max_steps: int = 0  # Unlimited steps
