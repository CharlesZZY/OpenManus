"""
Math Worker - Specialized agent for mathematical problem solving.
"""

from app.agent.workers.base_knowledge_worker import BaseKnowledgeWorker


class MathWorker(BaseKnowledgeWorker):
    """
    Specialized worker for mathematical problem solving.
    
    Handles tasks like:
    - Mathematical reasoning and proofs
    - Formula derivation
    - Calculation problems
    - Statistical analysis
    """

    name: str = "MathWorker"
    description: str = (
        "专注于数学问题求解的 Agent，擅长数学推理、公式推导、计算题解答"
    )

    system_prompt: str = """你是一个数学专家 Agent。你的职责是：
1. 分析数学问题，识别问题类型（代数、几何、微积分、概率统计等）
2. 提供清晰的解题步骤和推理过程
3. 给出准确的计算结果
4. 必要时解释数学概念和公式

请用清晰、条理的方式回答问题，确保每个步骤都有解释。

输出格式：
- 问题分析：简述问题类型和关键信息
- 解题步骤：详细的推导过程
- 最终答案：明确的结论
- 验证（可选）：对答案的检验"""

    max_steps: int = 0  # Unlimited steps
