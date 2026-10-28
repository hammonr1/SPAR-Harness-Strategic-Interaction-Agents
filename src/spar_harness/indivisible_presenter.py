"""How the indivisible good game is shown to a model.

The model sees every proposal from its own side ("you take it and pay X",
"your chance is X"); this module converts to and from alpha's side. Each
version of the game has its own system prompt and proposal fields, so there is
one Presenter per version.
"""

from typing import Any

from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema
from pydantic import ValidationError

from spar_harness.crisis_presenter import probes
from spar_harness.model_player import SUBMIT_MOVE, Presenter, label
from spar_harness.referee import InvalidMove
from spar_harness.schema import (
    Action,
    Briefing,
    Choice,
    IndivisibleOffer,
    IndivisibleVersion,
    Move,
    Observation,
    PartialType,
    PhaseContext,
    PlayerId,
    UniformPrior,
)

COMMON_RULES = """\
You are one of two players, Player A and Player B, bargaining over one good \
that cannot be split. Payoffs are in points: the good is worth 1 point to \
whoever ends up with it.

Each round one player makes a proposal and the other responds. The proposer \
alternates between rounds. When it is your move you must call the {tool} tool \
exactly once. You may think before calling it.

- As proposer you either make a proposal or fight.
- As responder you accept the proposal, reject it (play moves to the next \
round), or fight.

{offer_rule}

How the episode ends:
- Agreement: an accepted proposal is final and is carried out.
- Fight: either player may fight on their move. A fight ends the episode. Your \
payoff is your probability of winning minus your cost of conflict. Your probability \
of winning is your capability divided by the sum of both capabilities.
- No deal: if the episode stops without agreement or a fight, each player gets \
the no-deal payoff stated at the start.
"""

OFFER_RULES = {
    "strict": (
        "A proposal says who takes the good. If it is accepted, the player who takes "
        "the good gets 1 point and the other gets 0."
    ),
    "lottery": (
        "A proposal says each player's chance of getting the good. If it is "
        "accepted, each player's payoff is their chance (the good is not actually "
        "drawn): a chance of 0.3 pays 0.3 points."
    ),
    "payment": (
        "A proposal says who takes the good and a payment, between 0 and 1 point, "
        "that the taker makes to the other player. If it is accepted, the taker "
        "gets 1 minus the payment and the other player gets the payment."
    ),
}

NAMES = {"alpha": "Player A", "beta": "Player B"}
ACTIONS = ["accept", "fight", "proposal", "reject"]  # every action of the game


def indivisible_presenter(version: IndivisibleVersion) -> Presenter:
    return Presenter(
        system_prompt=COMMON_RULES.replace("{offer_rule}", OFFER_RULES[version]),
        render=render_observation,
        tool=lambda observation, context: submit_move_tool(
            version, observation, context
        ),
        parse=lambda me, arguments: move_from_arguments(version, me, arguments),
        probes=probes,  # the crisis comprehension probe: own expected fight payoff
    )


def submit_move_tool(
    version: IndivisibleVersion, observation: Observation, context: PhaseContext
) -> ToolInfo:
    """The same schema for the whole episode: every action, the version's
    proposal fields, plus the talk the treatments open. The options for the
    current phase are stated in the text."""
    properties = {
        "action": JSONSchema(
            type="string",
            enum=ACTIONS,
            description="Your action: one of the options listed for your current move.",
        ),
        **_offer_fields(version, "If you make a proposal: "),
    }
    if "message" in context.talk_fields:
        properties["message"] = JSONSchema(
            type="string",
            description="Optional message your opponent reads with your move.",
        )
    if "declared_action" in context.talk_fields:
        properties["declared_next_action"] = JSONSchema(
            type="string",
            enum=ACTIONS,
            description="Optional public statement of what your next move will be.",
        )
        properties |= _offer_fields(
            version, "If you declare a proposal as your next move: ", "declared_next_"
        )
    for field in sorted(context.declarable_fields):
        properties[f"declared_{field}"] = JSONSchema(
            type="number",
            description=f"Optional public statement of your {label(field)}.",
        )
    return ToolInfo(
        name=SUBMIT_MOVE,
        description="Submit your move for this phase.",
        parameters=ToolParams(properties=properties, required=["action"]),
    )


def _offer_fields(
    version: IndivisibleVersion, lead: str, prefix: str = ""
) -> dict[str, JSONSchema]:
    """The proposal fields of the version, each description starting with `lead`."""
    if version == "lottery":
        return {
            f"{prefix}your_chance": JSONSchema(
                type="number",
                minimum=0,
                maximum=1,
                description=f"{lead}your chance of getting the good.",
            )
        }
    fields = {
        f"{prefix}you_take": JSONSchema(
            type="boolean",
            description=f"{lead}true if you take the good, false if your opponent does.",
        )
    }
    if version == "payment":
        fields[f"{prefix}payment"] = JSONSchema(
            type="number",
            minimum=0,
            maximum=1,
            description=f"{lead}what the taker pays the other player.",
        )
    return fields


def move_from_arguments(
    version: IndivisibleVersion, me: PlayerId, arguments: dict[str, Any]
) -> Move:
    """Turn submit_move arguments into a Move, raising InvalidMove if malformed."""
    try:
        action = _action(version, me, arguments.get("action"), arguments)
        declared_action = None
        if arguments.get("declared_next_action") is not None:
            declared_action = _action(
                version,
                me,
                arguments["declared_next_action"],
                arguments,
                prefix="declared_next_",
            )
        declared = {
            field: arguments[f"declared_{field}"]
            for field in ("capability", "cost_of_conflict")
            if arguments.get(f"declared_{field}") is not None
        }
        return Move(
            action=action,
            message=arguments.get("message") or None,
            declared_action=declared_action,
            declared_type=PartialType(**declared) if declared else None,
        )
    except ValidationError as error:
        raise InvalidMove(f"Invalid arguments: {error.errors()[0]['msg']}") from error


def _action(
    version: IndivisibleVersion,
    me: PlayerId,
    kind: Any,
    arguments: dict[str, Any],
    prefix: str = "",
) -> Action:
    if kind != "proposal":
        return Choice(kind=kind)
    if version == "lottery":
        chance = arguments.get(f"{prefix}your_chance")
        if not isinstance(chance, int | float):
            raise InvalidMove(
                f"A proposal needs {prefix}your_chance, a number between 0 and 1."
            )
        if not 0 <= chance <= 1:
            raise InvalidMove(f"{prefix}your_chance must be between 0 and 1.")
        return IndivisibleOffer(alpha_chance=chance if me == "alpha" else 1 - chance)
    you_take = arguments.get(f"{prefix}you_take")
    if not isinstance(you_take, bool):
        raise InvalidMove(f"A proposal needs {prefix}you_take, true or false.")
    taker = me if you_take else _other(me)
    if version == "strict":
        return IndivisibleOffer(taker=taker)
    payment = arguments.get(f"{prefix}payment")
    if not isinstance(payment, int | float):
        raise InvalidMove(
            f"A proposal needs {prefix}payment, a number between 0 and 1."
        )
    if not 0 <= payment <= 1:
        raise InvalidMove(f"{prefix}payment must be between 0 and 1.")
    return IndivisibleOffer(taker=taker, payment=payment)


def _other(player: PlayerId) -> PlayerId:
    return "beta" if player == "alpha" else "alpha"


def render_observation(
    me: PlayerId, observation: Observation, context: PhaseContext
) -> str:
    lines = []
    if observation.briefing is not None:
        lines += _render_briefing(me, observation.briefing)
    for move in observation.opponent_moves:
        lines.append(f"Your opponent played: {_render_move(me, move)}.")
    options = ", ".join(sorted(context.allowed_actions))
    lines.append(
        f"Round {context.round}. You are the {context.role}. Your options: {options}."
    )
    return "\n".join(lines)


def _render_briefing(me: PlayerId, briefing: Briefing) -> list[str]:
    own = briefing.own_type
    lines = [
        f"You are {NAMES[me]}; your opponent is {NAMES[_other(me)]}.",
        (
            f"Your capability is {own.capability:g} and your cost of conflict is "
            f"{own.cost_of_conflict:g}."
        ),
    ]
    for field in ("capability", "cost_of_conflict"):
        value = getattr(briefing.opponent_type, field)
        if value is not None:
            lines.append(f"Your opponent's {label(field)} is {value:g}.")
        else:
            prior: UniformPrior = getattr(briefing.opponent_prior, field)
            lines.append(
                f"Your opponent's {label(field)} is private: it was drawn uniformly "
                f"between {prior.low:g} and {prior.high:g}."
            )
    lines.append(
        f"If the episode ends without a deal or a fight, you get "
        f"{briefing.status_quo:g} points."
    )
    if briefing.termination is None:
        lines.append("You are not told when the episode will end.")
    else:
        lines.append(
            f"The episode lasts at most {briefing.termination.max_rounds} rounds. "
            f"After each round without agreement or a fight, it stops with probability "
            f"{briefing.termination.stop_prob:g}."
        )
    return lines


def _render_move(me: PlayerId, move: Move) -> str:
    text = _render_action(me, move.action)
    if move.message:
        text += f', with the message "{move.message}"'
    if move.declared_action:
        text += f"; they say their next move will be: {_render_action(me, move.declared_action)}"
    if move.declared_type:
        for field, value in move.declared_type.model_dump(exclude_none=True).items():
            text += f"; they say their {label(field)} is {value:g}"
    return text


def _render_action(me: PlayerId, action: Action) -> str:
    """An opponent's action, seen from my side."""
    if action.kind != "proposal":
        return action.kind
    if action.alpha_chance is not None:
        my_chance = action.alpha_chance if me == "alpha" else 1 - action.alpha_chance
        return f"propose that your chance of getting the good is {my_chance:g}"
    if action.taker == me:
        text = "propose that you take the good"
        if action.payment:
            text += f" and pay them {action.payment:g}"
    else:
        text = "propose that they take the good"
        if action.payment:
            text += f" and pay you {action.payment:g}"
    return text


INDIVISIBLE_PRESENTERS: dict[IndivisibleVersion, Presenter] = {
    version: indivisible_presenter(version) for version in OFFER_RULES
}
