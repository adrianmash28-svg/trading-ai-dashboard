from app.services.challenger_generator import generate_challenger_variants
from app.services.promotion_engine import promote_challenger, should_promote_challenger
from app.services.research_engine import ResearchEngine
from app.services.strategy_evaluator import EvaluationResult, build_paper_metrics_from_backtest, evaluate_strategy

__all__ = [
    "EvaluationResult",
    "ResearchEngine",
    "build_paper_metrics_from_backtest",
    "evaluate_strategy",
    "generate_challenger_variants",
    "promote_challenger",
    "should_promote_challenger",
]
