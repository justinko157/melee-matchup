"""When a newly trained model may take the champion alias."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    candidate_log_loss: float, baseline_log_loss: float, champion_log_loss: float | None
) -> PromotionDecision:
    """A candidate must beat the seed baseline, then match or beat the champion.

    All log losses are measured on the candidate's own test split.
    """
    if not candidate_log_loss < baseline_log_loss:
        return PromotionDecision(False, "below_baseline")
    if champion_log_loss is None:
        return PromotionDecision(True, "no_champion")
    if candidate_log_loss <= champion_log_loss:
        return PromotionDecision(True, "beats_or_ties_champion")
    return PromotionDecision(False, "worse_than_champion")
