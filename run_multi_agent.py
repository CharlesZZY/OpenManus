"""
Multi-Agent Flow Runner

This script demonstrates how to use the Multi-Agent system with:
- Coordinator for task orchestration
- Multiple specialized Workers
- Execution tracing with graph visualization

Usage:
    python run_multi_agent.py
    
Or with a specific prompt:
    python run_multi_agent.py --prompt "Your task here"
"""

import argparse
import asyncio
import time
from pathlib import Path

from app.agent.coordinator import Coordinator
from app.agent.workers import (
    # Tool-based Workers
    BrowserWorker,
    CodeWorker,
    FileWorker,
    SearchWorker,
    # Knowledge-based Workers
    MathWorker,
    CopywriterWorker,
    HistoryWorker,
    SummarizerWorker,
    # Hybrid Workers
    ResearchWorker,
    DataAnalystWorker,
)
from app.flow.multi_agent import MultiAgentFlow
from app.logger import logger


def create_workers():
    """Create all available worker agents."""
    return {
        # Tool-based Workers
        "browser": BrowserWorker(),
        "code": CodeWorker(),
        "file": FileWorker(),
        "search": SearchWorker(),
        # Knowledge-based Workers
        "math": MathWorker(),
        "copywriter": CopywriterWorker(),
        "history": HistoryWorker(),
        "summarizer": SummarizerWorker(),
        # Hybrid Workers
        "research": ResearchWorker(),
        "data_analyst": DataAnalystWorker(),
    }


def create_minimal_workers():
    """Create a minimal set of workers for basic tasks."""
    return {
        "search": SearchWorker(),
        "code": CodeWorker(),
        "file": FileWorker(),
        "math": MathWorker(),
        "copywriter": CopywriterWorker(),
        "summarizer": SummarizerWorker(),
    }


async def run_multi_agent(prompt: str, use_all_workers: bool = False):
    """
    Run the multi-agent system with the given prompt.
    
    Args:
        prompt: The user's task/request
        use_all_workers: Whether to use all workers or a minimal set
    """
    # Create workers
    if use_all_workers:
        workers = create_workers()
        logger.info(f"Created {len(workers)} workers (full set)")
    else:
        workers = create_minimal_workers()
        logger.info(f"Created {len(workers)} workers (minimal set)")

    # Create coordinator with unlimited steps
    coordinator = Coordinator(max_steps=0)  # 0 means unlimited

    # Create the multi-agent flow with tracing enabled
    flow = MultiAgentFlow(
        coordinator=coordinator,
        workers=workers,
        enable_trace=True,
        trace_output_dir="./traces",
        auto_save_trace=True,
    )

    logger.info("=" * 60)
    logger.info("Starting Multi-Agent Execution")
    logger.info("=" * 60)
    logger.info(f"Task: {prompt}")
    logger.info(f"Workers: {', '.join(workers.keys())}")
    logger.info("=" * 60)

    try:
        start_time = time.time()
        
        # Execute the flow
        result = await flow.execute(prompt)
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("Execution Complete")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info("")
        
        # Print execution summary
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(flow.get_execution_summary())
        
        # Print result
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(result)
        
        # Print trace info
        if flow.enable_trace:
            summary = flow.get_trace_summary()
            print("\n" + "=" * 60)
            print("TRACE SUMMARY")
            print("=" * 60)
            print(f"Total nodes: {summary.get('total_nodes', 0)}")
            print(f"Total edges: {summary.get('total_edges', 0)}")
            print(f"Has cycles: {summary.get('has_cycles', False)}")
            print(f"Duration: {summary.get('duration_ms', 0):.0f}ms")
            
            # Print Mermaid diagram
            print("\n" + "=" * 60)
            print("MERMAID DIAGRAM")
            print("=" * 60)
            print(flow.get_trace_mermaid())
            
            # Save trace files
            saved_files = flow.save_trace()
            print("\n" + "=" * 60)
            print("SAVED TRACE FILES")
            print("=" * 60)
            for fmt, path in saved_files.items():
                print(f"  {fmt}: {path}")

        return result

    except asyncio.TimeoutError:
        logger.error("Execution timed out")
        return "Error: Execution timed out"
    except KeyboardInterrupt:
        logger.info("Execution cancelled by user")
        return "Cancelled by user"
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Multi-Agent System")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="The task prompt to execute"
    )
    parser.add_argument(
        "--all-workers", "-a",
        action="store_true",
        help="Use all available workers (default: minimal set)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600)"
    )
    
    args = parser.parse_args()
    
    # Get prompt from argument or interactive input
    if args.prompt:
        prompt = args.prompt
    else:
        print("=" * 60)
        print("Multi-Agent System")
        print("=" * 60)
        print("Available Workers:")
        print("  - browser: Web browsing and interaction")
        print("  - code: Python code execution")
        print("  - file: File operations")
        print("  - search: Web searching")
        print("  - math: Mathematical problem solving")
        print("  - copywriter: Content writing")
        print("  - history: Historical questions")
        print("  - summarizer: Information summarization")
        print("  - research: Comprehensive research")
        print("  - data_analyst: Data analysis")
        print("=" * 60)
        prompt = input("Enter your task: ").strip()
        
    if not prompt:
        logger.warning("Empty prompt provided")
        return

    try:
        await asyncio.wait_for(
            run_multi_agent(prompt, args.all_workers),
            timeout=args.timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"Execution timed out after {args.timeout} seconds")


if __name__ == "__main__":
    asyncio.run(main())
