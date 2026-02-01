"""
Workers module - Specialized worker agents for multi-agent systems.

This module provides various worker agents categorized by their capabilities:

Tool-based Workers (依赖外部工具):
- BrowserWorker: Web browsing and page interaction
- CodeWorker: Python code execution
- FileWorker: File operations
- SearchWorker: Web searching

Knowledge-based Workers (依赖 LLM 能力):
- MathWorker: Mathematical problem solving
- CopywriterWorker: Content writing
- HistoryWorker: Historical question answering
- SummarizerWorker: Information summarization

Hybrid Workers (结合工具和 LLM):
- ResearchWorker: Comprehensive research
- DataAnalystWorker: Data analysis
"""

from app.agent.workers.base_knowledge_worker import BaseKnowledgeWorker

# Tool-based Workers
from app.agent.workers.browser_worker import BrowserWorker
from app.agent.workers.code_worker import CodeWorker
from app.agent.workers.file_worker import FileWorker
from app.agent.workers.search_worker import SearchWorker

# Knowledge-based Workers
from app.agent.workers.math_worker import MathWorker
from app.agent.workers.copywriter_worker import CopywriterWorker
from app.agent.workers.history_worker import HistoryWorker
from app.agent.workers.summarizer_worker import SummarizerWorker

# Hybrid Workers
from app.agent.workers.research_worker import ResearchWorker
from app.agent.workers.data_analyst_worker import DataAnalystWorker

__all__ = [
    # Base class
    "BaseKnowledgeWorker",
    # Tool-based Workers
    "BrowserWorker",
    "CodeWorker",
    "FileWorker",
    "SearchWorker",
    # Knowledge-based Workers
    "MathWorker",
    "CopywriterWorker",
    "HistoryWorker",
    "SummarizerWorker",
    # Hybrid Workers
    "ResearchWorker",
    "DataAnalystWorker",
]

# Worker registry for dynamic lookup
WORKER_REGISTRY = {
    # Tool-based
    "browser": BrowserWorker,
    "code": CodeWorker,
    "file": FileWorker,
    "search": SearchWorker,
    # Knowledge-based
    "math": MathWorker,
    "copywriter": CopywriterWorker,
    "history": HistoryWorker,
    "summarizer": SummarizerWorker,
    # Hybrid
    "research": ResearchWorker,
    "data_analyst": DataAnalystWorker,
}


def get_worker(worker_type: str):
    """
    Get a worker class by type name.
    
    Args:
        worker_type: The type of worker (e.g., "browser", "code", "math")
    
    Returns:
        The worker class, or None if not found
    """
    return WORKER_REGISTRY.get(worker_type.lower())


def create_worker(worker_type: str, **kwargs):
    """
    Create a worker instance by type name.
    
    Args:
        worker_type: The type of worker
        **kwargs: Additional arguments to pass to the worker constructor
    
    Returns:
        A new worker instance
    
    Raises:
        ValueError: If the worker type is not found
    """
    worker_cls = get_worker(worker_type)
    if worker_cls is None:
        available = ", ".join(WORKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown worker type: {worker_type}. Available types: {available}"
        )
    return worker_cls(**kwargs)


def create_all_workers(**kwargs):
    """
    Create instances of all available workers.
    
    Args:
        **kwargs: Additional arguments to pass to worker constructors
    
    Returns:
        Dictionary mapping worker type names to worker instances
    """
    return {
        worker_type: worker_cls(**kwargs)
        for worker_type, worker_cls in WORKER_REGISTRY.items()
    }
