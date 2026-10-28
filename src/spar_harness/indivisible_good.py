"""Rules of the indivisible good game: crisis bargaining over a good that
cannot be split.

The fight, the types and the timing are those of crisis bargaining. Only the
offer differs, in one of three versions: strict (who takes the good), lottery
(alpha's chance of getting it) or payment (who takes it and what it pays the
other). Without a deal each player gets the status quo.
"""

from spar_harness import crisis_bargaining
from spar_harness.crisis_bargaining import reservation_value, win_probability
from spar_harness.schema import (
    Briefing,
    Choice,
    Ending,
    EpisodeConfig,
    EpisodeState,
    GameSpec,
    IndivisibleOffer,
    IndivisibleVersion,
    Move,
    PhaseContext,
    PhaseSpec,
    PlayerId,
    PrivateType,
    StageSpec,
    TerminationSpec,
)

IN_RANGE_TOLERANCE = 1e-9  # a deal exactly at a fight value counts as in range


def deal_value(offer: IndivisibleOffer, player: PlayerId) -> float:
    """The player's payoff if the offer is accepted.

    A lottery pays its chance in expectation, like a fight. Otherwise the taker
    gets the good minus the payment and the other player gets the payment; in
    the strict version the payment is zero.
    """
    if offer.alpha_chance is not None:
        return offer.alpha_chance if player == "alpha" else 1 - offer.alpha_chance
    return 1 - offer.payment if player == offer.taker else offer.payment


def fight_values(types: dict[PlayerId, PrivateType]) -> dict[PlayerId, float]:
    """Each player's expected payoff from a fight, as in crisis bargaining."""
    p = win_probability(types["alpha"].capability, types["beta"].capability)
    return {
        player: reservation_value(player, p, types[player].cost_of_conflict)
        for player in ("alpha", "beta")
    }


def in_range(offer: IndivisibleOffer, types: dict[PlayerId, PrivateType]) -> bool:
    """Whether both players weakly prefer the deal to fighting."""
    fights = fight_values(types)
    return all(
        deal_value(offer, player) >= fights[player] - IN_RANGE_TOLERANCE
        for player in ("alpha", "beta")
    )


def outcome_payoffs(
    ending: Ending,
    types: dict[PlayerId, PrivateType],
    status_quo: float,
    offer: IndivisibleOffer | None = None,
) -> dict[PlayerId, float]:
    """Each player's payoff when the episode ends."""
    if ending == "fight":
        return fight_values(types)
    if ending == "agreement":
        return {player: deal_value(offer, player) for player in ("alpha", "beta")}
    return {"alpha": status_quo, "beta": status_quo}


def even_offer(version: IndivisibleVersion, proposer: PlayerId) -> IndivisibleOffer:
    """The offer that gives both players a half: the proposer claims the good
    (strict), a fair coin (lottery), or the proposer takes it and pays 0.5."""
    if version == "lottery":
        return IndivisibleOffer(alpha_chance=0.5)
    payment = 0.5 if version == "payment" else 0.0
    return IndivisibleOffer(taker=proposer, payment=payment)


def indivisible_good_game(
    version: IndivisibleVersion,
    max_rounds: int,
    stop_prob: float,
    status_quo: float = 0.0,
) -> GameSpec:
    """Alternating offers over the good: each round the proposer offers and the
    responder answers."""
    return GameSpec(
        game_id="indivisible_good",
        stages=[
            StageSpec(
                name="bargaining",
                phases=[
                    PhaseSpec(kind="act", actors="proposer", action_kind="offer"),
                    PhaseSpec(kind="act", actors="responder", action_kind="response"),
                ],
                termination=TerminationSpec(max_rounds=max_rounds, stop_prob=stop_prob),
                when_exhausted="horizon",
            )
        ],
        status_quo=status_quo,
        version=version,
    )


ALLOWED_ACTIONS = {
    "offer": {"proposal", "fight"},
    "response": {"accept", "reject", "fight"},
}


def ending_after_move(move: Move) -> Ending | None:
    if move.action.kind == "accept":
        return "agreement"
    if move.action.kind == "fight":
        return "fight"
    return None


def move_problem(game: GameSpec, move: Move) -> str | None:
    """A proposal must use exactly the fields of the game's version."""
    offer = move.action
    if offer is None or offer.kind != "proposal":
        return None
    version = game.version
    if version == "lottery":
        if offer.alpha_chance is None or offer.taker is not None:
            return "in the lottery version a proposal states alpha's chance only"
    elif offer.taker is None or offer.alpha_chance is not None:
        return f"in the {version} version a proposal states who takes the good"
    if version == "strict" and offer.payment != 0.0:
        return "in the strict version nothing is paid"
    return None


def episode_payoffs(game: GameSpec, state: EpisodeState) -> dict[PlayerId, float]:
    return outcome_payoffs(
        state.ending, state.types, game.status_quo, agreed_offer(state)
    )


def agreed_offer(state: EpisodeState) -> IndivisibleOffer | None:
    """The accepted proposal, or None without agreement."""
    if state.ending != "agreement":
        return None
    last_round = state.rounds[-1]
    return last_round.moves[last_round.proposer].action


def default_move(
    game: GameSpec, state: EpisodeState, player: PlayerId, context: PhaseContext
) -> Move:
    """The move made for a player who keeps failing. Never a fight."""
    if context.role == "responder":
        return Move(action=Choice(kind="reject"))
    for record in reversed(state.rounds):
        own_move = record.moves.get(player)
        if (
            record.proposer == player
            and own_move
            and own_move.action.kind == "proposal"
        ):
            return own_move
    return Move(action=even_offer(game.version, player))


def briefing(config: EpisodeConfig, state: EpisodeState, player: PlayerId) -> Briefing:
    """The crisis bargaining briefing, plus the version."""
    base = crisis_bargaining.briefing(config, state, player)
    return base.model_copy(update={"version": config.game.version})
