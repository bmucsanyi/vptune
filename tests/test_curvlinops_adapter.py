import pytest
import torch

import vptune as vp
from vptune import AdmissionError, Candidate, empirical_fisher_vp, fisher_vp, ggnvp, hvp
from vptune.adapters.curvlinops import (
    CurvLinOpsAdmitter,
    CurvLinOpsFisherMCSemantics,
    curvlinops_axis,
    curvlinops_operator_axis,
    curvlinops_runtime_config,
)


def monte_carlo_fisher(family: str, objective_id: str) -> vp.OperatorSpec:
    return fisher_vp(
        family,
        objective_id,
        aggregation="mean_per_example",
        distribution="categorical",
        label_policy="model_distribution",
        expectation="monte_carlo",
        sample_space="classes",
        loss_reduction="mean",
        denominator="num_examples",
        logits_axis=1,
        sample_count=3,
        seed=17,
    )


def fisher_mc_semantics(
    *,
    loss_function: str = "CrossEntropyLoss",
    seed: int = 17,
    sample_count: int = 3,
    loss_reduction: str = "mean",
    distribution: str = "categorical",
    label_policy: str = "model_distribution",
    sample_space: str = "classes",
    denominator: str = "num_examples",
    logits_axis: int | None = 1,
) -> CurvLinOpsFisherMCSemantics:
    return CurvLinOpsFisherMCSemantics(
        loss_function=loss_function,
        seed=seed,
        sample_count=sample_count,
        loss_reduction=loss_reduction,
        distribution=distribution,
        label_policy=label_policy,
        sample_space=sample_space,
        denominator=denominator,
        logits_axis=logits_axis,
    )


def test_curvlinops_axis_rejects_unknown_operator() -> None:
    axis = curvlinops_operator_axis(("hessian", "ggn", "fisher_mc", "empirical_fisher"))

    assert axis.settings_keys == ("curvlinops_operator",)

    with pytest.raises(AdmissionError):
        curvlinops_operator_axis(("unknown",))


def test_curvlinops_axis_applies_semantic_admission() -> None:
    axis = curvlinops_axis(
        operator=hvp("family", "loss", aggregation="sum"),
        allowed_operators=("hessian", "ggn"),
        loss_reduction="sum",
        aggregation_to_loss_reduction={"sum": "sum"},
    )

    assert axis.admit(
        Candidate("family", "hessian", {"curvlinops_operator": "hessian"})
    ) == (True, None)
    assert (
        axis.admit(Candidate("family", "ggn", {"curvlinops_operator": "ggn"}))[0]
        is False
    )


def test_curvlinops_axis_records_fisher_mc_semantics() -> None:
    semantics = fisher_mc_semantics()
    axis = curvlinops_axis(
        operator=monte_carlo_fisher("family", "loss"),
        allowed_operators=("fisher_mc",),
        loss_reduction="mean",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
        fisher_mc_semantics=semantics,
    )

    assert axis.admit(
        Candidate("family", "fisher", {"curvlinops_operator": "fisher_mc"})
    ) == (True, None)
    assert axis.signature()["identity"]["fisher_mc_semantics"] == {
        "loss_function": "CrossEntropyLoss",
        "seed": 17,
        "sample_count": 3,
        "loss_reduction": "mean",
        "distribution": "categorical",
        "label_policy": "model_distribution",
        "sample_space": "classes",
        "denominator": "num_examples",
        "logits_axis": 1,
    }


def test_curvlinops_admitter_rejects_operator_kind_mismatch() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=hvp("family", "loss", aggregation="sum"),
        allowed_operators=("hessian", "ggn"),
        loss_reduction="sum",
        aggregation_to_loss_reduction={"sum": "sum"},
    )
    rejected = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "ggn"})
    )

    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None


def test_curvlinops_admitter_rejects_aggregation_reduction_mismatch() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=hvp("family", "loss", aggregation="mean_per_example"),
        allowed_operators=("hessian",),
        loss_reduction="sum",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
    )
    rejected = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "hessian"})
    )

    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None


def test_curvlinops_admitter_accepts_matching_operator_and_reduction() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=ggnvp("family", "loss", aggregation="mean_per_example"),
        allowed_operators=("ggn",),
        loss_reduction="mean",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
    )
    accepted = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "ggn"})
    )

    assert accepted.admission_status == "passed"


def test_curvlinops_admitter_rejects_fisher_without_mc_semantics() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=monte_carlo_fisher("family", "loss"),
        allowed_operators=("fisher_mc",),
        loss_reduction="mean",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
    )
    rejected = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "fisher_mc"})
    )

    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None


def test_curvlinops_admitter_accepts_fisher_with_mc_semantics() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=monte_carlo_fisher("family", "loss"),
        allowed_operators=("fisher_mc",),
        loss_reduction="mean",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
        fisher_mc_semantics=fisher_mc_semantics(),
    )
    accepted = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "fisher_mc"})
    )

    assert accepted.admission_status == "passed"


def test_curvlinops_admitter_rejects_invalid_fisher_mc_semantics() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=monte_carlo_fisher("family", "loss"),
        allowed_operators=("fisher_mc",),
        loss_reduction="mean",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
        fisher_mc_semantics=fisher_mc_semantics(
            sample_count=0,
            loss_reduction="sum",
        ),
    )
    rejected = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "fisher_mc"})
    )

    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None


@pytest.mark.parametrize(
    ("field", "semantics"),
    [
        ("seed", fisher_mc_semantics(seed=18)),
        ("sample_count", fisher_mc_semantics(sample_count=4)),
        ("distribution", fisher_mc_semantics(distribution="empirical")),
        ("label_policy", fisher_mc_semantics(label_policy="observed_labels")),
        ("sample_space", fisher_mc_semantics(sample_space="targets")),
        ("denominator", fisher_mc_semantics(denominator="one")),
        ("logits_axis", fisher_mc_semantics(logits_axis=0)),
    ],
)
def test_curvlinops_admitter_rejects_fisher_mc_semantic_mismatch(
    field: str,
    semantics: CurvLinOpsFisherMCSemantics,
) -> None:
    admitter = CurvLinOpsAdmitter(
        operator=monte_carlo_fisher("family", "loss"),
        allowed_operators=("fisher_mc",),
        loss_reduction="mean",
        aggregation_to_loss_reduction={"mean_per_example": "mean"},
        fisher_mc_semantics=semantics,
    )
    rejected = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "fisher_mc"})
    )

    assert rejected.admission_status == "failed"
    assert rejected.admission_error is not None
    assert field in rejected.admission_error


def test_curvlinops_admitter_accepts_empirical_fisher() -> None:
    admitter = CurvLinOpsAdmitter(
        operator=empirical_fisher_vp("family", "loss", aggregation="sum"),
        allowed_operators=("empirical_fisher",),
        loss_reduction="sum",
        aggregation_to_loss_reduction={"sum": "sum"},
    )
    accepted = admitter.admit(
        Candidate("family", "row", {"curvlinops_operator": "empirical_fisher"})
    )

    assert accepted.admission_status == "passed"


def test_curvlinops_runtime_config_records_parameter_order_identity() -> None:
    class FakeLinearOperator:
        def __matmul__(self, vector: object) -> object:
            return vector

    def operator_factory(
        candidate: Candidate,
        batch: vp.Batch,
    ) -> FakeLinearOperator:
        assert candidate
        assert batch

        return FakeLinearOperator()

    def reference_check(
        candidate: Candidate,
        batch: vp.Batch,
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate
        assert batch
        assert vector

        return vp.ReferenceResult(
            "curvlinops",
            {"max_abs_diff": 1e-6},
            {"max_abs_diff": 0.0},
        )

    def materializer_impl(
        candidate: Candidate,
        record: vp.FullSizeRecord,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == record.candidate_id

        return vp.constant_operation(torch.tensor([1.0]))

    materializer = vp.CallableMaterializer(
        "tests.curvlinops.materializer",
        "1",
        {},
        materializer_impl,
    )

    runtime = curvlinops_runtime_config(
        operator=hvp("family", "loss", aggregation="sum"),
        operator_factory=operator_factory,
        operator_factory_id="fake-linear-operator",
        candidates=(Candidate("family", "row", {}, admission_status="passed"),),
        reference_check=reference_check,
        materializer=materializer,
        parameter_names=("b", "a"),
        axis_registry=None,
        identity={"loss_reduction": "sum"},
    )

    assert runtime.identity()["parameter_names"] == ("b", "a")
    assert runtime.identity()["operator_factory_id"] == "fake-linear-operator"
