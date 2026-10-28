from pytest import approx

from spar_harness.indivisible_good import (
    deal_value,
    even_offer,
    fight_values,
    in_range,
    indivisible_good_game,
    move_problem,
    outcome_payoffs,
)
from spar_harness.schema import Choice, IndivisibleOffer, Move, PrivateType

TYPES = {
    "alpha": PrivateType(capability=3.0, cost_of_conflict=0.1),
    "beta": PrivateType(capability=1.0, cost_of_conflict=0.2),
}  # p = 0.75, fight values 0.65 and 0.05


def test_strict_deal_gives_the_good_to_the_taker():
    offer = IndivisibleOffer(taker="beta")
    assert deal_value(offer, "alpha") == 0.0
    assert deal_value(offer, "beta") == 1.0


def test_lottery_deal_pays_each_chance_in_expectation():
    offer = IndivisibleOffer(alpha_chance=0.7)
    assert deal_value(offer, "alpha") == approx(0.7)
    assert deal_value(offer, "beta") == approx(0.3)


def test_payment_deal_moves_the_payment_from_the_taker_to_the_other():
    offer = IndivisibleOffer(taker="alpha", payment=0.3)
    assert deal_value(offer, "alpha") == approx(0.7)
    assert deal_value(offer, "beta") == approx(0.3)
    offer = IndivisibleOffer(taker="beta", payment=0.3)
    assert deal_value(offer, "alpha") == approx(0.3)
    assert deal_value(offer, "beta") == approx(0.7)


def test_fight_values_are_the_crisis_reservation_values():
    assert fight_values(TYPES) == {"alpha": approx(0.65), "beta": approx(0.05)}


def test_agreement_pays_the_deal_values():
    payoffs = outcome_payoffs(
        "agreement", TYPES, status_quo=0.0, offer=IndivisibleOffer(alpha_chance=0.8)
    )
    assert payoffs == {"alpha": approx(0.8), "beta": approx(0.2)}


def test_fight_pays_the_fight_values():
    payoffs = outcome_payoffs("fight", TYPES, status_quo=0.5)
    assert payoffs == {"alpha": approx(0.65), "beta": approx(0.05)}


def test_no_deal_pays_the_status_quo_to_each_player():
    for ending in ("horizon", "random_stop"):
        assert outcome_payoffs(ending, TYPES, status_quo=0.0) == {
            "alpha": 0.0,
            "beta": 0.0,
        }
        assert outcome_payoffs(ending, TYPES, status_quo=0.5) == {
            "alpha": 0.5,
            "beta": 0.5,
        }


def test_in_range_means_both_prefer_the_deal_to_fighting():
    assert in_range(IndivisibleOffer(alpha_chance=0.7), TYPES)
    assert in_range(IndivisibleOffer(alpha_chance=0.65), TYPES)  # at the edge
    assert not in_range(IndivisibleOffer(alpha_chance=0.6), TYPES)
    assert in_range(IndivisibleOffer(taker="alpha", payment=0.1), TYPES)
    assert not in_range(IndivisibleOffer(taker="alpha", payment=0.4), TYPES)


def test_strict_has_no_deal_both_prefer_to_fighting_when_fight_values_are_positive():
    assert all(value > 0 for value in fight_values(TYPES).values())
    for taker in ("alpha", "beta"):
        assert not in_range(IndivisibleOffer(taker=taker), TYPES)


def test_strict_has_a_deal_when_one_fight_value_is_not_positive():
    types = {
        "alpha": PrivateType(capability=9.0, cost_of_conflict=0.1),
        "beta": PrivateType(capability=1.0, cost_of_conflict=0.2),
    }  # beta's fight value is -0.1
    assert in_range(IndivisibleOffer(taker="alpha"), types)
    assert not in_range(IndivisibleOffer(taker="beta"), types)


def test_even_offer_gives_both_players_a_half():
    assert even_offer("strict", "beta") == IndivisibleOffer(taker="beta")
    assert even_offer("lottery", "beta") == IndivisibleOffer(alpha_chance=0.5)
    assert even_offer("payment", "beta") == IndivisibleOffer(taker="beta", payment=0.5)
    assert deal_value(even_offer("lottery", "alpha"), "beta") == 0.5
    assert deal_value(even_offer("payment", "alpha"), "beta") == 0.5


def test_a_proposal_must_match_the_versions_fields():
    def problem(version, offer):
        game = indivisible_good_game(version, max_rounds=1, stop_prob=0.0)
        return move_problem(game, Move(action=offer))

    assert problem("strict", IndivisibleOffer(taker="alpha")) is None
    assert problem("strict", IndivisibleOffer(taker="alpha", payment=0.1))
    assert problem("strict", IndivisibleOffer(alpha_chance=0.5))
    assert problem("lottery", IndivisibleOffer(alpha_chance=0.5)) is None
    assert problem("lottery", IndivisibleOffer(taker="alpha"))
    assert problem("lottery", IndivisibleOffer(alpha_chance=0.5, taker="alpha"))
    assert problem("payment", IndivisibleOffer(taker="alpha", payment=0.1)) is None
    assert problem("payment", IndivisibleOffer(taker="alpha")) is None
    assert problem("payment", IndivisibleOffer(alpha_chance=0.5))
    assert problem("payment", IndivisibleOffer(taker="alpha", alpha_chance=0.5))


def test_choices_are_never_a_problem():
    game = indivisible_good_game("strict", max_rounds=1, stop_prob=0.0)
    assert move_problem(game, Move(action=Choice(kind="fight"))) is None
