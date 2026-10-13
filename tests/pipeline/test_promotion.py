from pipeline.ml.promotion import PromotionDecision, decide_promotion


def test_first_model_is_promoted():
    assert decide_promotion(0.50, 0.60, None) == PromotionDecision(True, "no_champion")


def test_better_model_is_promoted():
    assert decide_promotion(0.45, 0.60, 0.50) == PromotionDecision(True, "beats_or_ties_champion")


def test_equal_model_is_promoted():
    assert decide_promotion(0.50, 0.60, 0.50) == PromotionDecision(True, "beats_or_ties_champion")


def test_worse_model_is_not_promoted():
    assert decide_promotion(0.55, 0.60, 0.50) == PromotionDecision(False, "worse_than_champion")


def test_model_that_loses_to_the_baseline_is_never_promoted():
    assert decide_promotion(0.61, 0.60, None) == PromotionDecision(False, "below_baseline")
