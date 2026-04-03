"""Tests for the Elo rating engine."""


from src.features import DEFAULT_ELO, EloEngine


def test_initial_rating():
    elo = EloEngine()
    assert elo.get_rating(999) == DEFAULT_ELO


def test_expected_score_equal_ratings():
    elo = EloEngine()
    assert elo.expected_score(1500, 1500) == 0.5


def test_expected_score_higher_wins():
    elo = EloEngine()
    # A player 200 points higher should have >50% expected
    assert elo.expected_score(1700, 1500) > 0.5
    assert elo.expected_score(1500, 1700) < 0.5


def test_expected_score_400_gap():
    elo = EloEngine()
    # 400-point gap = ~10:1 expected ratio
    expected = elo.expected_score(1900, 1500)
    assert abs(expected - 10 / 11) < 0.01


def test_update_winner_gains_loser_loses():
    elo = EloEngine()
    elo.update(1, 2)
    assert elo.get_rating(1) > DEFAULT_ELO
    assert elo.get_rating(2) < DEFAULT_ELO


def test_update_conserves_roughly():
    """Rating changes should roughly balance (not exactly due to dynamic K)."""
    elo = EloEngine()
    # After many games, dynamic K converges to base K, so changes balance
    for _ in range(50):
        elo.update(1, 2)
        elo.update(2, 1)
    # Both should be close to default after equal wins
    assert abs(elo.get_rating(1) - DEFAULT_ELO) < 50
    assert abs(elo.get_rating(2) - DEFAULT_ELO) < 50


def test_upset_causes_bigger_swing():
    """A lower-rated player beating a higher-rated one should cause a bigger change."""
    elo1 = EloEngine()
    elo2 = EloEngine()

    # Set up a rating gap: player 1 is strong, player 2 is weak
    for _ in range(30):  # get past dynamic K phase
        elo1.update(10, 11)
        elo2.update(10, 11)

    r_strong_before = elo1.get_rating(10)

    # Expected win: strong beats weak
    elo1.update(10, 11)
    expected_gain = elo1.get_rating(10) - r_strong_before

    r_strong_before_2 = elo2.get_rating(10)

    # Upset: weak beats strong
    elo2.update(11, 10)
    upset_loss = r_strong_before_2 - elo2.get_rating(10)

    # Upset should cause a bigger rating change than an expected result
    assert upset_loss > expected_gain


def test_games_played_tracked():
    elo = EloEngine()
    elo.update(1, 2)
    elo.update(1, 3)
    assert elo.games_played[1] == 2
    assert elo.games_played[2] == 1
    assert elo.games_played[3] == 1
