"""
Copywriter Worker - Specialized agent for content writing.
"""

from app.agent.workers.base_knowledge_worker import BaseKnowledgeWorker


class CopywriterWorker(BaseKnowledgeWorker):
    """
    Specialized worker for content writing and copywriting.
    
    Handles tasks like:
    - Marketing copy
    - Article writing
    - Product descriptions
    - Email composition
    - Social media content
    """

    name: str = "CopywriterWorker"
    description: str = (
        "专注于文案创作的 Agent，擅长撰写各类文案、文章、营销内容"
    )

    system_prompt: str = """你是一个专业文案写作 Agent。你的职责是：
1. 根据需求撰写高质量的文案内容
2. 适应不同的写作风格（正式、轻松、专业、创意等）
3. 针对目标受众优化语言和表达
4. 确保内容结构清晰、逻辑通顺

你可以创作的内容包括：
- 营销文案和广告语
- 产品描述和介绍
- 文章和博客
- 邮件和通讯
- 社交媒体内容

写作原则：
- 抓住读者注意力
- 传达清晰的核心信息
- 使用恰当的语气和风格
- 注意标点和格式"""

    max_steps: int = 0  # Unlimited steps
