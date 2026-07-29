"""Evaluation framework for RAG agent."""
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel

from .agent.graph import rag_graph
from .agent.llm import LLMClient, create_llm_client
from .agent.config import settings


class EvalCase(BaseModel):
    """Single evaluation case."""
    question: str
    expected_answer: str
    expected_sources: List[str] = []
    category: str = "general"


class EvalResult(BaseModel):
    """Result of a single evaluation."""
    question: str
    expected: str
    actual: str
    sources: List[Dict[str, Any]]
    score: float
    passed: bool
    category: str
    judge_reasoning: str


class EvalSuite(BaseModel):
    """Complete evaluation suite results."""
    results: List[EvalResult]
    summary: Dict[str, Any]


JUDGE_PROMPT = """You are an expert evaluator. Compare the expected answer with the actual answer and score from 0.0 to 1.0.

Criteria:
- 1.0: Perfect match, all key information present and accurate
- 0.8: Minor omissions or slight inaccuracies
- 0.5: Major omissions or significant inaccuracies
- 0.2: Barely relevant, mostly wrong
- 0.0: Completely wrong or unrelated

Question: {question}
Expected Answer: {expected}
Actual Answer: {actual}

Respond with JSON:
{{
    "score": 0.0-1.0,
    "reasoning": "brief explanation",
    "passed": true/false
}}"""


class RAGEvaluator:
    """Evaluator for RAG system."""
    
    def __init__(self, judge_client: Optional[LLMClient] = None):
        self.judge_client = judge_client
    
    async def _get_judge(self) -> LLMClient:
        if self.judge_client:
            return self.judge_client
        return create_llm_client(settings)
    
    async def evaluate_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single test case."""
        # Run RAG
        result = await rag_graph.ainvoke({"question": case.question})
        
        actual = result.get("answer", "")
        sources = result.get("sources", [])
        error = result.get("error")
        
        if error:
            return EvalResult(
                question=case.question,
                expected=case.expected_answer,
                actual=f"ERROR: {error}",
                sources=[],
                score=0.0,
                passed=False,
                category=case.category,
                judge_reasoning="RAG system error",
            )
        
        # Judge evaluation
        judge = await self._get_judge()
        
        prompt = JUDGE_PROMPT.format(
            question=case.question,
            expected=case.expected_answer,
            actual=actual,
        )
        
        messages = [
            {"role": "system", "content": "You are an expert evaluator. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = await judge.complete(messages)
            await judge.close()
            
            # Parse JSON from response
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                judge_data = json.loads(json_match.group())
            else:
                judge_data = {"score": 0.0, "reasoning": "Failed to parse judge response", "passed": False}
        except Exception as e:
            judge_data = {"score": 0.0, "reasoning": f"Judge error: {e}", "passed": False}
        
        return EvalResult(
            question=case.question,
            expected=case.expected_answer,
            actual=actual,
            sources=sources,
            score=judge_data.get("score", 0.0),
            passed=judge_data.get("passed", False),
            category=case.category,
            judge_reasoning=judge_data.get("reasoning", ""),
        )
    
    async def evaluate_suite(self, cases: List[EvalCase]) -> EvalSuite:
        """Evaluate multiple cases."""
        results = []
        for case in cases:
            print(f"Evaluating: {case.question[:50]}...")
            result = await self.evaluate_case(case)
            results.append(result)
            print(f"  Score: {result.score:.2f} {'✓' if result.passed else '✗'}")
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / total if total > 0 else 0
        
        by_category = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = {"total": 0, "passed": 0, "scores": []}
            by_category[r.category]["total"] += 1
            by_category[r.category]["passed"] += 1 if r.passed else 0
            by_category[r.category]["scores"].append(r.score)
        
        cat_summary = {}
        for cat, data in by_category.items():
            cat_summary[cat] = {
                "pass_rate": data["passed"] / data["total"],
                "avg_score": sum(data["scores"]) / len(data["scores"]),
            }
        
        summary = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_score": avg_score,
            "by_category": cat_summary,
        }
        
        return EvalSuite(results=results, summary=summary)


# Default eval cases
DEFAULT_EVAL_CASES = [
    EvalCase(
        question="What is this project about?",
        expected_answer="This is a RAG agent starter kit using LangGraph, ChromaDB, and Ollama for local LLM inference.",
        category="general",
    ),
    EvalCase(
        question="What components does the project use?",
        expected_answer="The project uses LangGraph for orchestration, ChromaDB for vector storage, and Ollama for local LLM inference.",
        category="technical",
    ),
    EvalCase(
        question="How do I run the ingestion pipeline?",
        expected_answer="Run the ingest_pipeline function or call the /ingest API endpoint to load and index documents.",
        category="usage",
    ),
]


async def run_evals(cases: Optional[List[EvalCase]] = None, output_path: Optional[str] = None) -> EvalSuite:
    """Run evaluation suite."""
    cases = cases or DEFAULT_EVAL_CASES
    evaluator = RAGEvaluator()
    suite = await evaluator.evaluate_suite(cases)
    
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(suite.model_dump(), f, indent=2)
        print(f"Results saved to {output_path}")
    
    print(f"\n=== EVAL SUMMARY ===")
    print(f"Total: {suite.summary['total']}")
    print(f"Passed: {suite.summary['passed']}")
    print(f"Pass Rate: {suite.summary['pass_rate']:.1%}")
    print(f"Avg Score: {suite.summary['avg_score']:.2f}")
    
    return suite


if __name__ == "__main__":
    asyncio.run(run_evals(output_path="eval_results.json"))