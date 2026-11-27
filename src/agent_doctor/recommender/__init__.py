"""Recommender: fix plans grounded in a library of playbooks (retrieval-augmented).

>>> from agent_doctor.recommender import recommend
>>> plan = recommend(report, run)                       # deterministic, free
>>> plan = recommend(report, run, llm=llm_from_spec("anthropic"))  # plus an LLM-written plan
"""

from agent_doctor.recommender.knowledge import KnowledgeBase, Playbook, default_knowledge_base
from agent_doctor.recommender.llm import llm_from_spec
from agent_doctor.recommender.recommend import (
    FixPlan,
    Recommendation,
    build_prompt,
    recommend,
    recommend_many,
    run_providers,
)

__all__ = [
    "FixPlan",
    "KnowledgeBase",
    "Playbook",
    "Recommendation",
    "build_prompt",
    "default_knowledge_base",
    "llm_from_spec",
    "recommend",
    "recommend_many",
    "run_providers",
]
