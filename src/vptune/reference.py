"""Reference-check helpers."""

import dataclasses
from collections.abc import Mapping

from vptune.checks import tree_error_measurements, validate_thresholds
from vptune.data import (
    Batch,
    Candidate,
    CandidateOperation,
    OperationFactory,
    ReferenceCheck,
    ReferenceResult,
)
from vptune.errors import MaterializationError
from vptune.tensor_tree import TensorTree


def tree_reference_check(
    *,
    anchor_factory: OperationFactory,
    candidate_factory: OperationFactory,
    thresholds: Mapping[str, float],
    anchor_candidate_id: str,
) -> ReferenceCheck:
    """Return a reference check that compares a candidate to an anchor."""

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        if anchor_candidate_id == candidate.candidate_id:
            message = "reference anchor candidate_id must differ from candidate_id"
            raise MaterializationError(message)

        anchor_candidate = dataclasses.replace(
            candidate,
            candidate_id=anchor_candidate_id,
        )
        anchor_operation = anchor_factory(anchor_candidate, batch, vector)
        candidate_operation = candidate_factory(candidate, batch, vector)
        measurements = tree_error_measurements(
            candidate_operation(),
            anchor_operation(),
        )
        validate_thresholds(measurements, thresholds)

        return ReferenceResult(
            name="tree_close",
            thresholds=dict(thresholds),
            measurements=measurements,
        )

    return check


def constant_operation(value: TensorTree) -> CandidateOperation:
    """Return an operation that always produces the provided tensor tree."""

    def operation() -> TensorTree:
        return value

    return operation
