from fastapi import APIRouter

from shared.state import (
    get_strategy_lab_activity_snapshot,
    get_strategy_lab_experiments_snapshot,
    get_strategy_lab_paper_tests_snapshot,
    get_strategy_lab_promotions_snapshot,
    get_strategy_lab_summary_snapshot,
)


router = APIRouter(prefix="/strategy-lab", tags=["strategy-lab"])


@router.get("/summary")
def strategy_lab_summary():
    return get_strategy_lab_summary_snapshot()


@router.get("/activity")
def strategy_lab_activity(limit: int = 25):
    return get_strategy_lab_activity_snapshot(limit=limit)


@router.get("/experiments")
def strategy_lab_experiments(limit: int = 50):
    return get_strategy_lab_experiments_snapshot(limit=limit)


@router.get("/paper-tests")
def strategy_lab_paper_tests(limit: int = 25):
    return get_strategy_lab_paper_tests_snapshot(limit=limit)


@router.get("/promotions")
def strategy_lab_promotions(limit: int = 25):
    return get_strategy_lab_promotions_snapshot(limit=limit)
