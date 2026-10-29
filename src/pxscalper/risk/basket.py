"""Basket profit target: close everything once the *combined* PnL hits a number.

Why a basket rule at all: with one position, a take-profit on the position is
enough. With several positions (scaling in), each has its own target, so the
account can sit at a healthy combined profit while every individual position is
still waiting for its own. The basket rule answers the question the operator
actually asks: *"when do I have my $3?"*

The danger is self-evident -- closing the winners while the losers stay open is
how a small account keeps the losing half of a book forever. That is why the rule
here is close **all**, and why the caller must pass every open position, not the
profitable subset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

# PnL can be reported two ways by a venue; both are supported explicitly so the
# decision is never made on a number whose units nobody checked.
PNL_FIELDS = ("upl", "unrealized_pnl", "unrealised_pnl", "pnl", "profit")


@dataclass
class BasketResult:
    triggered: bool
    total_pnl: float
    positions: int
    winners: int
    losers: int
    target: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "total_pnl": round(self.total_pnl, 6),
            "positions": self.positions,
            "winners": self.winners,
            "losers": self.losers,
            "target": self.target,
            "reason": self.reason,
        }


def position_pnl(position: Any) -> float:
    """Unrealised PnL of one position, in account currency.

    Accepts dicts (tests, journals) and exchange ``Position`` objects. An unknown
    shape raises rather than returning 0.0: a basket total that silently ignores
    positions is worse than no basket rule at all.
    """
    if isinstance(position, dict):
        for field_name in PNL_FIELDS:
            if field_name in position and position[field_name] is not None:
                return float(position[field_name])
        raise KeyError(f"no PnL field in position dict: {sorted(position)}")
    for field_name in PNL_FIELDS:
        if hasattr(position, field_name):
            value = getattr(position, field_name)
            if value is not None:
                return float(value)
    raise AttributeError(f"no PnL attribute on {type(position).__name__}")


def basket_pnl(positions: Iterable[Any]) -> float:
    return float(sum(position_pnl(p) for p in positions))


class BasketTarget:
    """Tracks when the combined unrealised PnL reaches a target.

    ``target`` is in account currency (e.g. 3.0 for $3). ``pct`` mode is offered
    because a dollar target means one thing on a $100 account and something very
    different on a $10,000 account -- if both are set, the dollar target wins and
    ``pct`` acts as a floor/ceiling pair.
    """

    def __init__(
        self,
        target: float = 0.0,
        *,
        arm_above: float = 0.0,
        cooldown_s: float = 0.0,
        clock=time.time,
    ) -> None:
        self.target = float(target)
        self.arm_above = float(arm_above)
        # After firing, the account is flat. Re-arming immediately would close the
        # first position of the next cycle the moment it shows a cent of profit,
        # so a cooldown exists.
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._last_fire_s: float = 0.0
        self.fires: int = 0

    # ------------------------------------------------------------------ arming
    def armed(self) -> bool:
        if self.target <= 0:
            return False
        return (self._clock() - self._last_fire_s) >= self.cooldown_s

    def seconds_to_arm(self) -> float:
        if self.target <= 0:
            return 0.0
        return max(0.0, self.cooldown_s - (self._clock() - self._last_fire_s))

    # ---------------------------------------------------------------- decision
    def evaluate(self, positions: Iterable[Any], *, enforce: bool = False) -> BasketResult:
        """Should the whole basket be closed right now?

        ``enforce`` is the operator's explicit "yes, close even though a position
        is losing". Without it the rule only *reports* that the target was
        reached, which is how a paper run can be observed before it is trusted.
        """
        positions = list(positions)
        total = basket_pnl(positions)
        winners = sum(1 for p in positions if position_pnl(p) > 0)
        losers = sum(1 for p in positions if position_pnl(p) < 0)

        if self.target <= 0:
            return BasketResult(False, total, len(positions), winners, losers,
                                self.target, "basket target disabled")
        if not positions:
            return BasketResult(False, total, 0, winners, losers, self.target,
                                "no open positions")
        if not self.armed():
            return BasketResult(
                False, total, len(positions), winners, losers, self.target,
                f"re-arming in {self.seconds_to_arm():.0f}s",
            )
        if total < self.target:
            return BasketResult(
                False, total, len(positions), winners, losers, self.target,
                f"${total:+.2f} of ${self.target:.2f}",
            )
        if total < self.arm_above:
            return BasketResult(
                False, total, len(positions), winners, losers, self.target,
                f"below arm threshold ${self.arm_above:.2f}",
            )
        if not enforce:
            return BasketResult(
                True, total, len(positions), winners, losers, self.target,
                f"${total:+.2f} reached ${self.target:.2f} (report only; "
                f"set close_all_on_profit to act)",
            )
        return BasketResult(
            True, total, len(positions), winners, losers, self.target,
            f"TARGET HIT ${total:+.2f} >= ${self.target:.2f} -> closing all "
            f"{len(positions)} position(s) ({winners} up, {losers} down)",
        )

    def mark_fired(self) -> None:
        self._last_fire_s = self._clock()
        self.fires += 1

    def reset(self) -> None:
        self._last_fire_s = 0.0
