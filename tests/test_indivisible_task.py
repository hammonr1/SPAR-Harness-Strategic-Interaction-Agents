"""End to end: a scripted mock model plays the indivisible good through Inspect."""

import math

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.indivisible_good import fight_values
from spar_harness.indivisible_presenter import indivisible_presenter
from spar_harness.indivisible_task import indivisible_good
from spar_harness.schema import (
    Briefing,
    EpisodeState,
    IndivisibleOffer,
    Move,
    Observation,
    PartialType,
    PhaseContext,
    PrivateType,
)

VERSIONS = ("strict", "lottery", "payment")

PROPOSALS = {  # what the mock proposes: it takes the good, worth 0.6 to it
    "strict": {"action": "proposal", "you_take": True},
    "lottery": {"action": "proposal", "your_chance": 0.6},
    "payment": {"action": "proposal", "you_take": True, "payment": 0.4},
}


def last_user_text(input):
    """What the mock reads to decide, like a real model: the latest prompt."""
    return [message for message in input if message.role == "user"][-1].text


def is_probe(input):
    return "private question" in last_user_text(input).lower()


def scripted_model(version, probe_leaks):
    """Answers the probe, proposes to take the good when proposing, accepts otherwise."""

    def generate(input, tools, tool_choice, config):
        if is_probe(input):
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.4})
        if "private question" in str([message.model_dump() for message in input]):
            probe_leaks.append(input)
        if "You are the proposer" in last_user_text(input):
            return ModelOutput.for_tool_call("mock", "submit_move", PROPOSALS[version])
        return ModelOutput.for_tool_call("mock", "submit_move", {"action": "accept"})

    return generate


@pytest.mark.parametrize("version", VERSIONS)
def test_model_plays_a_pair_of_episodes_against_the_bot(version):
    probe_leaks = []
    [log] = eval(
        indivisible_good(n_episodes=1, a="model", b="indivisible_bot", version=version),
        model_roles={
            "a": get_model(
                "mockllm/model", custom_outputs=scripted_model(version, probe_leaks)
            )
        },
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    assert len(log.samples) == 2  # one pair: seat a plays alpha, then beta
    for sample in log.samples:
        episode = sample.store["episode"]
        assert episode["ending"] in {"agreement", "fight", "horizon", "random_stop"}
        assert all(not record["defaulted"] for record in episode["rounds"])
        scores = sample.scores["indivisible_outcome_scorer"].value
        assert set(scores) == {
            "fight",
            "agreement",
            "rounds_to_agreement",
            "efficiency",
            "in_range",
        }
        assert sample.scores["seat_scorer"].value["a_comprehension"] in (0, 1)
    # the probe was a side call: the model never saw it again when moving
    assert probe_leaks == []


def test_model_recovers_from_an_invalid_move_with_a_well_formed_conversation():
    malformed_problems = []
    fumbled = []

    def fumbling_model(input, tools, tool_choice, config):
        # every tool call the model made earlier must be answered by a tool result
        for message, following in zip(input, input[1:] + [None]):
            answered = following is not None and following.role == "tool"
            if getattr(message, "tool_calls", None) and not answered:
                malformed_problems.append(message)
        if is_probe(input):
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.4})
        if "You are the proposer" in last_user_text(input):
            if not fumbled:
                fumbled.append(True)  # a payment proposal without its payment
                return ModelOutput.for_tool_call(
                    "mock", "submit_move", {"action": "proposal", "you_take": True}
                )
            return ModelOutput.for_tool_call(
                "mock", "submit_move", PROPOSALS["payment"]
            )
        return ModelOutput.for_tool_call("mock", "submit_move", {"action": "accept"})

    [log] = eval(
        indivisible_good(
            n_episodes=1, a="model", b="indivisible_bot", version="payment"
        ),
        model_roles={"a": get_model("mockllm/model", custom_outputs=fumbling_model)},
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    assert fumbled
    assert malformed_problems == []
    for sample in log.samples:
        assert all(
            not record["defaulted"] for record in sample.store["episode"]["rounds"]
        )


def test_payment_bots_at_status_quo_one_half_reproduce_crisis_bargaining():
    [log] = eval(
        indivisible_good(
            n_episodes=3,
            a="indivisible_bot",
            b="indivisible_bot",
            version="payment",
            status_quo=0.5,
        ),
        model="mockllm/model",
        display="none",
        log_dir="logs/tests",
    )
    metrics = {
        score.name: score.metrics["spar_harness/mean_of_defined"].value
        for score in log.results.scores
    }
    # complete information: every episode agrees in round 1, inside the range
    assert metrics["agreement"] == 1.0
    assert metrics["rounds_to_agreement"] == 1.0
    assert metrics["in_range"] == 1.0
    assert metrics["efficiency"] == 1.0
    # bots answer no probes and declare no actions: undefined, not zero
    assert math.isnan(metrics["a_comprehension"])
    assert math.isnan(metrics["b_intention_gap"])


def test_strict_bots_fight_exactly_when_both_fight_values_are_positive():
    [log] = eval(
        indivisible_good(
            n_episodes=5, a="indivisible_bot", b="indivisible_bot", version="strict"
        ),
        model="mockllm/model",
        display="none",
        log_dir="logs/tests",
    )
    endings = set()
    for sample in log.samples:
        episode = EpisodeState.model_validate(sample.store["episode"])
        both_positive = all(value > 0 for value in fight_values(episode.types).values())
        assert episode.ending == ("fight" if both_positive else "agreement"), sample.id
        endings.add(episode.ending)
    assert endings == {"fight", "agreement"}  # the default priors produce both


@pytest.mark.parametrize("version", VERSIONS)
def test_the_tools_never_change_within_a_conversation(version):
    [log] = eval(
        indivisible_good(
            n_episodes=1,
            a="model",
            b="indivisible_bot",
            version=version,
            messages=True,
            declared_action=True,
        ),
        model_roles={
            "a": get_model("mockllm/model", custom_outputs=scripted_model(version, []))
        },
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    for sample in log.samples:
        tool_lists = [
            [tool.model_dump() for tool in event.tools]
            for event in sample.events
            if event.event == "model"
        ]
        assert len(tool_lists) > 1
        assert all(tools == tool_lists[0] for tools in tool_lists), sample.id


def briefing(version):
    return Briefing(
        own_type=PrivateType(capability=3.0, cost_of_conflict=0.1),
        opponent_type=PartialType(capability=1.0, cost_of_conflict=0.2),
        opponent_prior=None,
        private_fields=set(),
        status_quo=0.0,
        termination=None,
        version=version,
    )


def test_the_presenter_shows_proposals_from_the_players_own_side():
    context = PhaseContext(
        round=2, role="responder", action_kind="response", allowed_actions={"accept"}
    )

    def shown(version, me, offer):
        observation = Observation(
            recipient=me, round=2, opponent_moves=[Move(action=offer)]
        )
        return indivisible_presenter(version).render(me, observation, context)

    assert "they take the good" in shown(
        "strict", "alpha", IndivisibleOffer(taker="beta")
    )
    assert "you take the good" in shown(
        "strict", "beta", IndivisibleOffer(taker="beta")
    )
    assert "your chance of getting the good is 0.3" in shown(
        "lottery", "beta", IndivisibleOffer(alpha_chance=0.7)
    )
    assert "they take the good and pay you 0.2" in shown(
        "payment", "beta", IndivisibleOffer(taker="alpha", payment=0.2)
    )
    assert "you take the good and pay them 0.2" in shown(
        "payment", "alpha", IndivisibleOffer(taker="alpha", payment=0.2)
    )


def test_the_presenter_parses_proposals_to_alphas_side():
    parse = indivisible_presenter("payment").parse
    move = parse("beta", {"action": "proposal", "you_take": True, "payment": 0.2})
    assert move.action == IndivisibleOffer(taker="beta", payment=0.2)
    move = parse("beta", {"action": "proposal", "you_take": False, "payment": 0.2})
    assert move.action == IndivisibleOffer(taker="alpha", payment=0.2)
    parse = indivisible_presenter("lottery").parse
    move = parse("beta", {"action": "proposal", "your_chance": 0.3})
    assert move.action.alpha_chance == pytest.approx(0.7)
    parse = indivisible_presenter("strict").parse
    move = parse("alpha", {"action": "proposal", "you_take": False})
    assert move.action == IndivisibleOffer(taker="beta")


def test_the_briefing_names_the_players_and_the_no_deal_payoff():
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"proposal"}
    )
    observation = Observation(recipient="beta", round=1, briefing=briefing("strict"))
    text = indivisible_presenter("strict").render("beta", observation, context)
    assert "You are Player B; your opponent is Player A." in text
    assert "without a deal or a fight, you get 0 points" in text
    assert "Your options: proposal." in text
