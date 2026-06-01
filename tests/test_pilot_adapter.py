import dataclasses
from collections.abc import Mapping

import pytest
import torch

import vptune as vp
from vptune.adapters.pilot import (
    acceptance_family_names,
    acceptance_readiness,
    lower,
    readiness,
    require_acceptance_families,
    selected_settings,
    validators,
)
from vptune.schemas import compute_record_owner_hash


class OneBatchData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "pilot"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {"family": family, "check": check_name}

    @staticmethod
    def probe_batches(family: str) -> tuple[Mapping[str, object], ...]:
        return ({"family": family},)


class OneVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "pilot-vector"}

    @staticmethod
    def reference_vectors(family: str) -> torch.Tensor:
        assert family

        return torch.tensor([1.0])

    @staticmethod
    def probe_vectors(family: str) -> tuple[torch.Tensor, ...]:
        assert family

        return (torch.tensor([1.0]),)


def cpu_target() -> vp.Target:
    return vp.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float32",),
        allowed_attention_impls=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=vp.TimingPolicy(),
        selection_policy=vp.SelectionPolicy(),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )


def reference_passed() -> vp.ReferenceResult:
    return vp.ReferenceResult(
        "tree_close",
        {"max_abs_diff": 1e-6},
        {"max_abs_diff": 0.0},
    )


def materialize_candidate_impl(
    candidate: vp.Candidate,
    record: vp.FullSizeRecord,
) -> vp.CandidateOperation:
    assert record.candidate_id == candidate.candidate_id

    return vp.constant_operation(torch.tensor([1.0]))


materialize_candidate = vp.CallableMaterializer(
    "tests.pilot.materialize_candidate",
    "1",
    {},
    materialize_candidate_impl,
)


def problem_for(name: str, operator: vp.OperatorSpec) -> vp.Problem:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate(
        name,
        f"{name}:row",
        {"axis": name},
        admission_status="passed",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.family == name
        assert batch["family"] == name
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.family == name
        assert batch["family"] == name

        return vp.constant_operation(vector)

    return vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=vp.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": name},
        ),
    )


def full_size_record(candidate: vp.Candidate) -> vp.FullSizeRecord:
    sample = vp.Measurement(
        elapsed_seconds=1.0,
        peak_allocated_mib=1.0,
        peak_reserved_mib=1.0,
        post_allocated_mib=0.0,
        post_reserved_mib=0.0,
    )

    record = vp.FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature={"case": "pilot"},
        candidate_settings=candidate.settings,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="full_size",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            input_signature={"case": "pilot"},
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        timing_samples=(sample,),
        memory_samples=(sample,),
        dependency_identities=dict(candidate.dependency_identities),
    )

    return dataclasses.replace(record, content_hash=record.computed_content_hash())


def plan_for(candidate: vp.Candidate) -> vp.Plan:
    record = full_size_record(candidate)

    return vp.Plan(
        selected={candidate.family: candidate},
        records={candidate.family: record},
        input_signature={"case": "pilot"},
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        materializers={candidate.family: materialize_candidate},
        validation_order=(candidate.family,),
    )


def test_pilot_lower_validates_family_problem_match() -> None:
    target = cpu_target()
    first_operator = vp.gradient("first", "loss", aggregation="sum")
    second_operator = vp.hvp("second", "loss", aggregation="sum")
    run = lower(
        target=target,
        families=(
            vp.Family("second", second_operator, dependencies=("first",)),
            vp.Family("first", first_operator),
        ),
        problems=(
            problem_for("first", first_operator),
            problem_for("second", second_operator),
        ),
        run_id="pilot",
    )

    assert tuple(family.name for family in run.families) == ("first", "second")

    with pytest.raises(vp.MaterializationError):
        lower(
            target=target,
            families=(vp.Family("first", first_operator),),
            problems=(problem_for("second", second_operator),),
            run_id="bad",
        )


def test_pilot_readiness_and_selected_settings() -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"axis": "value"},
        admission_status="passed",
    )
    plan = plan_for(candidate)
    state = readiness(plan, ("family",))
    settings = selected_settings(plan, ("family",))
    stale = dataclasses.replace(
        plan,
        records={
            "family": dataclasses.replace(
                plan.records["family"],
                status="failed",
            )
        },
    )

    assert state.passed()
    assert settings["family"]["candidate_id"] == "row"
    assert settings["family"]["settings"] == {"axis": "value"}
    assert not readiness(stale, ("family",)).passed()

    with pytest.raises(vp.MaterializationError):
        selected_settings(stale, ("family",))


def test_pilot_selected_settings_require_validation_rows_when_plan_requires_them() -> (
    None
):
    candidate = vp.Candidate(
        "family",
        "row",
        {"axis": "value"},
        admission_status="passed",
    )
    plan = dataclasses.replace(
        plan_for(candidate),
        validation_required=True,
        validation_order=("family",),
        validator_identities={"family": {"validator": "test"}},
    )

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.family == "family"
        assert record.family == "family"
        assert context.family == "family"

        return reference_passed()

    validation_records = vp.validate_plan(plan, {"family": validator})
    settings = selected_settings(
        plan,
        ("family",),
        validation_records=validation_records,
    )
    failed_record = dataclasses.replace(validation_records[0], status="failed")

    assert settings["family"]["candidate_id"] == "row"

    with pytest.raises(vp.MaterializationError, match="validation rows"):
        selected_settings(plan, ("family",))

    with pytest.raises(vp.MaterializationError, match="did not pass"):
        selected_settings(
            plan,
            ("family",),
            validation_records=(failed_record,),
        )


def test_pilot_acceptance_helpers_require_acceptance_family_set() -> None:
    required = acceptance_family_names()
    candidate = vp.Candidate(
        required[0],
        "row",
        {"axis": "value"},
        admission_status="passed",
    )
    plan = plan_for(candidate)
    state = acceptance_readiness(plan)

    require_acceptance_families(required)

    assert required == (
        "capability_gradient",
        "retain_kl_backward",
        "kfac_metric",
        "capability_hvp",
        "hessian_ritz",
        "chart_retain_curvature",
        "contact_training_step",
    )
    assert not state.passed()
    assert state.missing_families == required[1:]

    with pytest.raises(vp.MaterializationError):
        require_acceptance_families(("capability_gradient",))


def test_pilot_validators_require_exact_family_coverage() -> None:
    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.family == record.family
        assert context.family == candidate.family

        return reference_passed()

    accepted = validators(("family",), {"family": validator})

    assert tuple(accepted) == ("family",)

    with pytest.raises(vp.MaterializationError):
        validators(("family",), {})
