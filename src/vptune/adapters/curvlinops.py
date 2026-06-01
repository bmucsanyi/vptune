"""CurvLinOps adapter helpers."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch

from vptune.candidates import AxisDescriptor
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    Candidate,
    CandidateAdmitter,
    CandidateOperation,
    Materializer,
    OperationFactory,
    OperatorSpec,
    ReferenceCheck,
    RuntimeConfig,
)
from vptune.errors import AdmissionError
from vptune.tensor_tree import TensorTree

CURVLINOPS_OPERATOR_KIND = {
    "hessian": "hvp",
    "ggn": "ggnvp",
    "fisher_mc": "fisher_vp",
    "empirical_fisher": "empirical_fisher_vp",
}


class CurvLinOpsOperator(Protocol):
    """CurvLinOps-style linear operator."""

    def __matmul__(self, vector: Any) -> Any:
        """Return the matrix-free product."""


class CurvLinOpsFactory(Protocol):
    """Build a CurvLinOps-style operator for one candidate and batch."""

    def __call__(self, candidate: Candidate, batch: Batch) -> CurvLinOpsOperator:
        """Return a linear operator."""


@dataclasses.dataclass(frozen=True, slots=True)
class CurvLinOpsFisherMCSemantics:
    """Declared semantics for a CurvLinOps FisherMC row."""

    loss_function: str
    seed: int
    sample_count: int
    loss_reduction: str
    distribution: str
    label_policy: str
    sample_space: str
    denominator: str
    logits_axis: int | None = None

    def signature(self) -> dict[str, Any]:
        """Return stable FisherMC identity."""
        return {
            "loss_function": self.loss_function,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "loss_reduction": self.loss_reduction,
            "distribution": self.distribution,
            "label_policy": self.label_policy,
            "sample_space": self.sample_space,
            "denominator": self.denominator,
            "logits_axis": self.logits_axis,
        }

    def validation_error(self) -> str | None:
        """Return a validation error for invalid FisherMC semantics."""
        errors = []

        if not self.loss_function:
            errors.append("FisherMC loss_function must be explicit")

        if self.sample_count < 1:
            errors.append("FisherMC sample_count must be positive")

        required_string_fields = (
            "loss_reduction",
            "distribution",
            "label_policy",
            "sample_space",
            "denominator",
        )
        errors.extend(
            f"FisherMC {field_name} must be explicit"
            for field_name in required_string_fields
            if not getattr(self, field_name)
        )

        if errors:
            return "; ".join(errors)

        return None


def curvlinops_operator_axis(operators: Sequence[str]) -> AxisDescriptor:
    """Return a CurvLinOps operator-choice axis.

    Raises:
        AdmissionError: If an operator name is unsupported.
    """
    unsupported = tuple(
        operator for operator in operators if operator not in CURVLINOPS_OPERATOR_KIND
    )

    if unsupported:
        message = f"unsupported CurvLinOps operators: {unsupported}"
        raise AdmissionError(message)

    return AxisDescriptor(
        name="curvlinops_operator",
        settings_keys=("curvlinops_operator",),
        allowed_values=tuple(operators),
        adapter_id="vptune.curvlinops",
        adapter_version=PACKAGE_VERSION,
    )


def curvlinops_axis(
    *,
    operator: OperatorSpec,
    allowed_operators: Sequence[str],
    loss_reduction: str,
    aggregation_to_loss_reduction: Mapping[str, str],
    fisher_mc_semantics: CurvLinOpsFisherMCSemantics | None = None,
) -> AxisDescriptor:
    """Return a CurvLinOps axis with semantic admission attached."""
    value_axis = curvlinops_operator_axis(tuple(allowed_operators))
    admitter = CurvLinOpsAdmitter(
        operator=operator,
        allowed_operators=tuple(allowed_operators),
        loss_reduction=loss_reduction,
        aggregation_to_loss_reduction=dict(aggregation_to_loss_reduction),
        fisher_mc_semantics=fisher_mc_semantics,
    )

    def admission_rule(candidate: Candidate) -> tuple[bool, str | None]:
        admitted = admitter.admit(candidate)

        if admitted.admission_status == "passed":
            return True, None

        return False, admitted.admission_error

    return dataclasses.replace(
        value_axis,
        admission_rule=admission_rule,
        identity=admitter.signature(),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CurvLinOpsAdmitter:
    """Admit CurvLinOps candidates for one operator spec."""

    operator: OperatorSpec
    allowed_operators: tuple[str, ...]
    loss_reduction: str
    aggregation_to_loss_reduction: Mapping[str, str]
    fisher_mc_semantics: CurvLinOpsFisherMCSemantics | None = None

    def signature(self) -> dict[str, Any]:
        """Return stable CurvLinOps admission identity."""
        fisher_mc_semantics = (
            None
            if self.fisher_mc_semantics is None
            else self.fisher_mc_semantics.signature()
        )

        return {
            "adapter_id": "vptune.curvlinops",
            "adapter_version": PACKAGE_VERSION,
            "operator": self.operator.signature(),
            "allowed_operators": self.allowed_operators,
            "loss_reduction": self.loss_reduction,
            "aggregation_to_loss_reduction": dict(self.aggregation_to_loss_reduction),
            "fisher_mc_semantics": fisher_mc_semantics,
        }

    def admit(self, candidate: Candidate) -> Candidate:
        """Return candidate with CurvLinOps admission applied."""
        if candidate.admission_status == "failed":
            return candidate

        reason = self._admission_error(candidate)

        if reason is None:
            return dataclasses.replace(candidate, admission_status="passed")

        return dataclasses.replace(
            candidate,
            admission_status="failed",
            admission_error=reason,
        )

    def _admission_error(self, candidate: Candidate) -> str | None:
        errors = []
        operator_name = candidate.settings.get("curvlinops_operator")

        if not isinstance(operator_name, str):
            errors.append("curvlinops_operator must be a string")
            operator_name = ""
        elif operator_name not in self.allowed_operators:
            errors.append(f"CurvLinOps operator is not allowed: {operator_name}")

        if isinstance(operator_name, str) and operator_name:
            expected_kind = CURVLINOPS_OPERATOR_KIND.get(operator_name)

            if expected_kind is None:
                errors.append(f"CurvLinOps operator is unsupported: {operator_name}")
            elif self.operator.kind != expected_kind:
                errors.append(
                    f"CurvLinOps operator {operator_name} does not implement "
                    f"{self.operator.kind}"
                )
            elif operator_name == "fisher_mc":
                fisher_error = self._fisher_mc_error()

                if fisher_error is not None:
                    errors.append(fisher_error)

        reduction = self.aggregation_to_loss_reduction.get(self.operator.aggregation)

        if reduction is None:
            errors.append(
                "operator aggregation has no loss reduction: "
                f"{self.operator.aggregation}"
            )
        elif reduction != self.loss_reduction:
            errors.append(
                f"operator aggregation {self.operator.aggregation} expects "
                f"{reduction}, not {self.loss_reduction}"
            )

        if errors:
            return "; ".join(errors)

        return None

    def _fisher_mc_error(self) -> str | None:
        if self.fisher_mc_semantics is None:
            return "FisherMC requires explicit Monte Carlo Fisher semantics"

        validation_error = self.fisher_mc_semantics.validation_error()

        if validation_error is not None:
            return validation_error

        if self.fisher_mc_semantics.loss_reduction != self.loss_reduction:
            return (
                "FisherMC loss_reduction differs from adapter loss_reduction: "
                f"{self.fisher_mc_semantics.loss_reduction} != {self.loss_reduction}"
            )

        semantic_error = self._fisher_mc_semantic_mismatch()

        if semantic_error is not None:
            return semantic_error

        return None

    def _fisher_mc_semantic_mismatch(self) -> str | None:
        semantics = self.fisher_mc_semantics

        if semantics is None:
            return "FisherMC requires explicit Monte Carlo Fisher semantics"

        expected = {
            "distribution": semantics.distribution,
            "label_policy": semantics.label_policy,
            "expectation": "monte_carlo",
            "sample_space": semantics.sample_space,
            "loss_reduction": semantics.loss_reduction,
            "denominator": semantics.denominator,
            "sample_count": semantics.sample_count,
            "seed": semantics.seed,
        }

        if semantics.logits_axis is not None:
            expected["logits_axis"] = semantics.logits_axis

        for key, value in expected.items():
            if self.operator.semantics.get(key) != value:
                return f"FisherMC semantic field mismatch: {key}"

        return None


def curvlinops_operation_factory(
    operator_factory: CurvLinOpsFactory,
    parameter_names: Sequence[str] = (),
) -> OperationFactory:
    """Return an operation factory for CurvLinOps-style linear operators."""

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        def operation() -> TensorTree:
            operator = operator_factory(candidate, batch)
            curvlinops_vector, output_format = _to_curvlinops_vector(
                vector,
                parameter_names,
            )
            product = operator @ curvlinops_vector

            return _from_curvlinops_output(product, output_format)

        return operation

    return factory


def curvlinops_runtime_config(
    *,
    operator: OperatorSpec,
    operator_factory: CurvLinOpsFactory,
    operator_factory_id: str,
    candidates: Sequence[Candidate],
    reference_check: ReferenceCheck,
    materializer: Materializer,
    parameter_names: Sequence[str],
    axis_registry: CandidateAdmitter | None,
    identity: Mapping[str, Any],
) -> RuntimeConfig:
    """Return a CurvLinOps runtime config with adapter identity."""
    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=curvlinops_operation_factory(
            operator_factory,
            parameter_names=parameter_names,
        ),
        reference_check=reference_check,
        materializer=materializer,
        axis_registry=axis_registry,
        signature={
            "adapter_id": "vptune.curvlinops",
            "adapter_version": PACKAGE_VERSION,
            "operator": operator.signature(),
            "operator_factory_id": operator_factory_id,
            "parameter_names": tuple(parameter_names),
            "vector_format": "mapping" if parameter_names else "flat_or_sequence",
            "identity": dict(identity),
        },
    )


def _to_curvlinops_vector(
    vector: TensorTree,
    parameter_names: Sequence[str],
) -> tuple[torch.Tensor | list[torch.Tensor], str | tuple[str, ...]]:
    if isinstance(vector, torch.Tensor):
        return vector, "tensor"

    if isinstance(vector, tuple):
        return [_tensor_leaf(value) for value in vector], "tuple"

    if isinstance(vector, dict):
        names = tuple(parameter_names)

        if not names:
            message = "CurvLinOps dict vectors require explicit parameter_names"
            raise TypeError(message)

        if set(vector) != set(names):
            message = "CurvLinOps dict vector keys differ from parameter_names"
            raise RuntimeError(message)

        return [_tensor_leaf(vector[name]) for name in names], names

    message = "CurvLinOps adapter accepts tensor, tuple, or ordered dict vectors"
    raise TypeError(message)


def _tensor_leaf(value: TensorTree) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value

    message = "CurvLinOps tensor-list vectors must be flat tuples"
    raise TypeError(message)


def _from_curvlinops_output(
    value: Any,
    output_format: str | tuple[str, ...],
) -> TensorTree:
    if isinstance(value, torch.Tensor):
        if output_format != "tensor":
            message = "CurvLinOps output format differs from input vector format"
            raise TypeError(message)

        return value

    if isinstance(value, list):
        return _from_sequence_output(value, output_format)

    if isinstance(value, tuple):
        return _from_sequence_output(value, output_format)

    message = f"unsupported CurvLinOps output: {type(value).__name__}"
    raise TypeError(message)


def _from_sequence_output(
    values: Sequence[Any],
    output_format: str | tuple[str, ...],
) -> TensorTree:
    tensors = tuple(_tensor_leaf(item) for item in values)

    if output_format == "tuple":
        return tensors

    if isinstance(output_format, tuple):
        if len(tensors) != len(output_format):
            message = "CurvLinOps output length differs from parameter_names"
            raise RuntimeError(message)

        return dict(zip(output_format, tensors, strict=True))

    message = "CurvLinOps output format differs from input vector format"
    raise TypeError(message)
