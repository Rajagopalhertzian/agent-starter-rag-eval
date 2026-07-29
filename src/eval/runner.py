"""Evaluation harness with LLM-as-judge."""
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from ..agent.llm import OpenAICompatibleClient
from ..agent.config import settings


class EvalCase(BaseModel):
    """Single evaluation case."""
    question: str
    expected_answer: str
    expected_sources: Optional[List[str]] = None


class EvalResult(BaseModel):
    """Result of a single evaluation."""
    question: str
    expected: str
    actual: str
    faithfulness_score: float  # 1-5
    relevance_score: float     # 1-5
    passed: bool


FAITHFULNESS_PROMPT = """You are an evaluator. Rate the faithfulness of the answer to the provided context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Rate 1-5 where:
1 = Answer contradicts context or hallucinates
2 = Answer has major inaccuracies vs context
3 = Answer is partially accurate but missing key details
4 = Answer is mostly accurate with minor omissions
5 = Answer is fully supported by context

Output ONLY the number (1-5)."""


RELEVANCE_PROMPT = """You are an evaluator. Rate the relevance of the answer to the question.

Question:
{question}

Answer:
{answer}

Rate 1-5 where:
1 = Answer is completely irrelevant
2 = Answer barely addresses the question
3 = Answer partially addresses the question
4 = Answer mostly addresses the question
5 = Answer fully and directly addresses the question

Output ONLY the number (1-5)."""


class LLMJudge:
    """LLM-as-judge evaluator using local Ollama."""
    
    def __init__(self, model: str = None, api_base: str = None, api_key: str = None):
        self.client = OpenAICompatibleClient(
            api_base=api_base or settings.eval_judge_api_base,
            api_key=api_key or settings.eval_judge_api_key,
            model=model or settings.eval_judge_model,
            temperature=0.0,
        )
    
    async def judge_faithfulness(self, question: str, context: str, answer: str) -> int:
        prompt = FAITHFULNESS_PROMPT.format(
            context=context[:3000],  # truncate
            question=question,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]
        result = await self.client.complete(messages)
        await self.client.close()
        try:
            return int(result.strip())
        except ValueError:
            return 3  # default middle
    
    async def judge_relevance(self, question: str, answer: str) -> int:
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]
        result = await self.client.complete(messages)
        await self.client.close()
        try:
            return int(result.strip())
        except ValueError:
            return 3
    
    async def close(self):
        await self.client.close()


async def run_evaluation(
    test_cases: List[EvalCase],
    faithfulness_threshold: float = 3.0,
    relevance_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Run evaluation suite."""
    from ..agent.graph import rag_graph, RAGState
    
    judge = LLMJudge()
    results = []
    
    for case in test_cases:
        # Run RAG
        initial_state: RAGState = {
            "question": case.question,
            "context": [],
            "answer": "",
            "sources": [],
            "error": None,
        }
        
        result = await rag_graph.ainvoke(initial_state)
        actual_answer = result.get("answer", "")
        context = "\n\n".join([doc.page_content for doc in result.get("context", [])])
        
        # Judge
        faith = await judge.judge_faithfulness(case.question, context, actual_answer)
        relev = await judge.judge_relevance(case.question, actual_answer)
        
        passed = faith >= faithfulness_threshold and relev >= relevance_threshold
        
        results.append(EvalResult(
            question=case.question,
            expected=case.expected_answer,
            actual=actual_answer,
            faithfulness_score=faith,
            relevance_score=relev,
            passed=passed,
        ))
    
    await judge.close()
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_faith = sum(r.faithfulness_score for r in results) / total if total > 0 else 0
    avg_relev = sum(r.relevance_score for r in results) / total if total > 0 else 0
    
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total > 0 else 0,
        "avg_faithfulness": avg_faith,
        "avg_relevance": avg_relev,
        "results": [r.model_dump() for r in results],
    }


def load_test_cases(path: str) -> List[EvalCase]:
    """Load test cases from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [EvalCase(**item) for item in data]


if __name__ == "__main__":
    # Quick test
    cases = [
        EvalCase(
            question="What is this project?",
            expected_answer="A RAG agent starter kit with local LLMs",
        )
    ]
    result = asyncio.run(run_evaluation(cases))
    print(json.dumps(result, indent=2))