"""Bridge helpers for autobatch-backed candidate probes."""

from collections.abc import Callable, Hashable, Sequence
from typing import Protocol

import autobatch

from vptune.data import Candidate
from vptune.errors import MaterializationError


class AutobatchFind(Protocol):
    """Callable shape for `autobatch.find`."""

    def __call__(
        self,
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        """Run Autobatch search and return the selected value."""


def autobatch_goal(objective: str) -> autobatch.Goal:
    """Return the Autobatch goal selected by a domain objective.

    Raises:
        MaterializationError: If the goal name has no supported Autobatch mapping.
    """
    if objective == "largest_passing":
        return autobatch.Goal.largest_safe()

    if objective == "fastest_passing":
        return autobatch.Goal.fastest_step()

    message = f"autobatch objective is unsupported by this domain: {objective}"
    raise MaterializationError(message)


def find_autobatch_value(
    probe: Callable[[int], None],
    *,
    values: Sequence[int],
    objective: str,
    cache_key: Hashable,
    warmup_steps: int,
    measure_steps: int,
    devices: Sequence[int],
    find: AutobatchFind | None = None,
) -> int:
    """Run Autobatch over one integer domain and return the selected value.

    Returns:
        Selected integer value.
    """
    selected_find = autobatch.find if find is None else find

    return selected_find(
        probe,
        values=values,
        goal=autobatch_goal(objective),
        cache_key=cache_key,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        devices=list(devices),
    )


def select_fastest_candidate_with_autobatch(
    candidates: Sequence[Candidate],
    probe_candidate: Callable[[Candidate], None],
    *,
    cache_key: Hashable,
    warmup_steps: int,
    measure_steps: int,
    devices: list[int],
    find: AutobatchFind = autobatch.find,
) -> Candidate:
    """Return the fastest candidate selected by Autobatch.

    Raises:
        MaterializationError: If the candidate sequence is empty.
    """
    if not candidates:
        message = "autobatch candidate selection requires candidates"
        raise MaterializationError(message)

    values = tuple(range(1, len(candidates) + 1))

    def probe(value: int) -> None:
        probe_candidate(candidates[value - 1])

    selected_value = find_autobatch_value(
        probe,
        values=values,
        objective="fastest_passing",
        cache_key=cache_key,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        devices=devices,
        find=find,
    )

    return candidates[selected_value - 1]
