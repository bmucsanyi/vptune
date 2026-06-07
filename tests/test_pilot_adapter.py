import dataclasses
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

import vptune as vp
import vptune.ext as vpx
from vptune import operators as ops
from vptune.adapters.pilot import (
    PilotReadiness,
    lower,
    readiness,
    selected_settings,
)


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


def cpu_target() -> vpx.Target:
    return vpx.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("fp32",),
        allowed_attention_frontends=(),
        allowed_sdpa_kernels=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=vpx.TimingPolicy(),
        selection_policy=vpx.SelectionPolicy(),
        search_policy=vpx.SearchPolicy(strategy="exhaustive"),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )


def reference_passed() -> vpx.ReferenceResult:
    return vpx.ReferenceResult(
        "tree_close",
        {"max_abs_diff": 1e-6},
        {"max_abs_diff": 0.0},
    )


def materialize_candidate_impl(
    candidate: vpx.Candidate,
    record: vpx.FullSizeRecord,
) -> vpx.CandidateOperation:
    assert record.candidate_id == candidate.candidate_id

    return vpx.constant_operation(torch.tensor([1.0]))


materialize_candidate = vpx.CallableMaterializer(
    "tests.pilot.materialize_candidate",
    "1",
    {},
    {"callback": "tests.pilot.materialize_candidate_impl"},
    materialize_candidate_impl,
)


def problem_for(name: str, operator: vpx.OperatorSpec) -> vpx.Problem:
    model = torch.nn.Linear(1, 1)
    candidate = vpx.Candidate(
        name,
        f"{name}:row",
        {"axis": name},
        admission_status="passed",
    )

    def reference_check(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.ReferenceResult:
        assert candidate.family == name
        assert batch["family"] == name
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vpx.Candidate,
        batch: Mapping[str, object],
        vector: vpx.TensorTree,
    ) -> vpx.CandidateOperation:
        assert candidate.family == name
        assert batch["family"] == name

        return vpx.constant_operation(vector)

    wrapped_operation_factory = vpx.CallableOperationFactory(
        "tests.pilot.operation_factory",
        "1",
        {"generator": name},
        {"callback": "tests.pilot.operation_factory"},
        operation_factory,
    )
    wrapped_reference_check = vpx.CallableReferenceCheck(
        "tests.pilot.reference_check",
        "1",
        {"generator": name},
        {"callback": "tests.pilot.reference_check"},
        reference_check,
    )

    return vpx.Problem(
        model=model,
        params=vpx.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=vpx.RuntimeConfig(
            (candidate,),
            wrapped_operation_factory,
            wrapped_reference_check,
            materialize_candidate,
            None,
            {"generator": name},
        ),
    )


def full_size_record(candidate: vpx.Candidate) -> vpx.FullSizeRecord:
    sample = vpx.Measurement(
        elapsed_seconds=1.0,
        peak_allocated_mib=1.0,
        peak_reserved_mib=1.0,
        post_allocated_mib=0.0,
        post_reserved_mib=0.0,
    )

    record = vpx.FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature={"case": "pilot"},
        candidate_settings=candidate.settings,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        timing_samples=(sample,),
        memory_samples=(sample,),
        dependency_identities=dict(candidate.dependency_identities),
    )

    return record


def dependency_identity(
    candidate: vpx.Candidate,
    record: vpx.FullSizeRecord,
) -> dict[str, object]:
    return {
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "candidate_settings": dict(candidate.settings),
        "full_size_row": record.row_key(),
        "materializer_identity": dict(materialize_candidate.identity()),
    }


def refresh_check_record(record: vpx.CheckRecord) -> vpx.CheckRecord:
    return record


def plan_for(candidate: vpx.Candidate) -> vpx.Plan:
    record = full_size_record(candidate)

    return vpx.Plan(
        selected={candidate.family: candidate},
        records={candidate.family: record},
        input_signature={"case": "pilot"},
        policy=vpx.SelectionPolicy(),
        full_size_records=(record,),
        materializers={candidate.family: materialize_candidate},
        validation_order=(candidate.family,),
    )


def test_pilot_lower_validates_family_problem_match() -> None:
    target = cpu_target()
    first_operator = ops.gradient("first", "loss", aggregation="sum")
    second_operator = ops.hvp("second", "loss", aggregation="sum")
    constraint = vpx.CohortConstraint(
        name="dtype",
        settings_keys=("dtype.model_compute",),
        assignments=({"dtype.model_compute": "fp32"},),
        families=("first", "second"),
    )

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == record.family
        assert context.family in {"first", "second"}

        return reference_passed()

    run = lower(
        target=target,
        families=(
            vpx.Family("second", second_operator, dependencies=("first",)),
            vpx.Family("first", first_operator),
        ),
        problems=(
            problem_for("first", first_operator),
            problem_for("second", second_operator),
        ),
        run_id="pilot",
        cohort_constraints=(constraint,),
        plan_validators={"first": validator, "second": validator},
        validator_identities={
            "first": {"validator": "pilot"},
            "second": {"validator": "pilot"},
        },
    )

    assert tuple(family.name for family in run.families) == ("first", "second")
    assert run.cohort_constraints == (constraint,)
    assert set(run.validators) == {"first", "second"}
    assert run.validator_identities == {
        "first": {"validator": "pilot"},
        "second": {"validator": "pilot"},
    }

    with pytest.raises(vp.MaterializationError):
        lower(
            target=target,
            families=(vpx.Family("first", first_operator),),
            problems=(problem_for("second", second_operator),),
            run_id="bad",
        )

    with pytest.raises(vp.MaterializationError, match="validator identities"):
        lower(
            target=target,
            families=(vpx.Family("first", first_operator),),
            problems=(problem_for("first", first_operator),),
            run_id="bad-validators",
            plan_validators={"first": validator},
            validator_identities={},
        )


def test_pilot_readiness_and_selected_settings() -> None:
    candidate = vpx.Candidate(
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


def test_pilot_lowered_run_feeds_downstream_readiness_consumer(
    tmp_path: Path,
) -> None:
    operator = ops.gradient("family", "loss", aggregation="sum")

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == "family"
        assert record.family == "family"
        assert context.family == "family"

        return reference_passed()

    tuning = lower(
        target=cpu_target(),
        families=(vpx.Family("family", operator),),
        problems=(problem_for("family", operator),),
        run_id="pilot-e2e",
        plan_validators={"family": validator},
        validator_identities={"family": {"validator": "pilot-e2e"}},
    )
    plan = vp.tune_run(tuning, run_dir=tmp_path)
    state = readiness(plan, ("family",))
    settings = selected_settings(plan, ("family",))

    def downstream_consumer(
        ready: PilotReadiness,
        selected: Mapping[str, Mapping[str, object]],
    ) -> str:
        if not ready.passed():
            message = "pilot readiness did not pass"
            raise vp.MaterializationError(message)

        return str(selected["family"]["candidate_id"])

    assert state.passed()
    assert settings["family"]["candidate_id"] == "family:row"
    assert plan.validation_required
    assert tuple(record.status for record in plan.validation_records) == ("passed",)
    assert downstream_consumer(state, settings) == "family:row"


def test_pilot_readiness_rejects_stale_selected_candidate_metadata() -> None:
    candidate = vpx.Candidate(
        "family",
        "row",
        {"axis": "value"},
        admission_status="passed",
        generator_version="1",
    )
    plan = plan_for(candidate)
    changed_generator = dataclasses.replace(
        plan,
        selected={
            "family": dataclasses.replace(candidate, generator_version="2"),
        },
    )
    failed_admission = dataclasses.replace(
        plan,
        selected={
            "family": dataclasses.replace(
                candidate,
                admission_status="failed",
                admission_error="rejected",
            ),
        },
    )

    assert not readiness(changed_generator, ("family",)).passed()
    assert not readiness(failed_admission, ("family",)).passed()


def test_pilot_readiness_rejects_stale_selected_dependencies() -> None:
    dependency = vpx.Candidate(
        "dependency",
        "dependency-row",
        {"axis": "dependency"},
        admission_status="passed",
    )
    dependency_record = full_size_record(dependency)
    dependent = vpx.Candidate(
        "dependent",
        "dependent-row",
        {"axis": "dependent"},
        dependency_identities={
            "dependency": dependency_identity(dependency, dependency_record)
        },
        admission_status="passed",
    )
    plan = vpx.Plan(
        selected={
            "dependency": dependency,
            "dependent": dependent,
        },
        records={
            "dependency": dependency_record,
            "dependent": full_size_record(dependent),
        },
        input_signature={"case": "pilot"},
        policy=vpx.SelectionPolicy(),
        full_size_records=(dependency_record, full_size_record(dependent)),
        materializers={
            "dependency": materialize_candidate,
            "dependent": materialize_candidate,
        },
        validation_order=("dependency", "dependent"),
    )
    changed_dependent = dataclasses.replace(
        dependent,
        dependency_identities={
            "dependency": {
                **dependent.dependency_identities["dependency"],
                "candidate_id": "changed",
            }
        },
    )
    stale = dataclasses.replace(
        plan,
        selected={
            "dependency": dependency,
            "dependent": changed_dependent,
        },
    )

    assert not readiness(stale, ("dependent",)).passed()


def test_pilot_readiness_requires_selected_dependencies() -> None:
    dependency = vpx.Candidate(
        "dependency",
        "dependency-row",
        {"axis": "dependency"},
        admission_status="passed",
    )
    dependency_record = full_size_record(dependency)
    dependent = vpx.Candidate(
        "dependent",
        "dependent-row",
        {"axis": "dependent"},
        dependency_identities={
            "dependency": dependency_identity(dependency, dependency_record)
        },
        admission_status="passed",
    )
    dependent_record = full_size_record(dependent)
    missing_dependency_plan = vpx.Plan(
        selected={"dependent": dependent},
        records={"dependent": dependent_record},
        input_signature={"case": "pilot"},
        policy=vpx.SelectionPolicy(),
        full_size_records=(dependent_record,),
        materializers={"dependent": materialize_candidate},
        validation_order=("dependent",),
    )
    ready_plan = vpx.Plan(
        selected={"dependency": dependency, "dependent": dependent},
        records={
            "dependency": dependency_record,
            "dependent": dependent_record,
        },
        input_signature={"case": "pilot"},
        policy=vpx.SelectionPolicy(),
        full_size_records=(dependency_record, dependent_record),
        materializers={
            "dependency": materialize_candidate,
            "dependent": materialize_candidate,
        },
        validation_order=("dependency", "dependent"),
        dependencies_by_family={"dependent": ("dependency",)},
    )

    missing_state = readiness(missing_dependency_plan, ("dependent",))
    ready_state = readiness(ready_plan, ("dependent",))

    assert not missing_state.passed()
    assert missing_state.required_families == ("dependent", "dependency")
    assert missing_state.missing_families == ("dependency",)
    assert ready_state.passed()


def test_pilot_selected_settings_require_validation_rows_when_plan_requires_them() -> (
    None
):
    candidate = vpx.Candidate(
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
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == "family"
        assert record.family == "family"
        assert context.family == "family"

        return reference_passed()

    validation_records = vp.validate_plan(plan, {"family": validator})
    plan_with_records = dataclasses.replace(
        plan,
        validation_records=validation_records,
    )
    settings = selected_settings(
        plan,
        ("family",),
        validation_records=validation_records,
    )
    attached_settings = selected_settings(plan_with_records, ("family",))
    failed_record = dataclasses.replace(validation_records[0], status="failed")

    assert settings["family"]["candidate_id"] == "row"
    assert attached_settings["family"]["candidate_id"] == "row"

    with pytest.raises(vp.MaterializationError, match="validation rows"):
        selected_settings(plan, ("family",))

    with pytest.raises(vp.MaterializationError, match="did not pass"):
        selected_settings(
            plan,
            ("family",),
            validation_records=(failed_record,),
        )


def test_pilot_selected_settings_rejects_failed_plan_validator() -> None:
    candidate = vpx.Candidate(
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
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == "family"
        assert record.family == "family"
        assert context.selected().equal(torch.tensor([1.0]))

        message = "selected implementation failed"
        raise vp.ReferenceFailedError(message)

    with pytest.raises(vp.ReferenceFailedError, match="selected implementation"):
        vp.validate_plan(plan, {"family": validator})


def test_pilot_selected_settings_rejects_validation_dependency_mismatch() -> None:
    dependency = vpx.Candidate(
        "dependency",
        "dependency-row",
        {"axis": "dependency"},
        admission_status="passed",
    )
    dependency_record = full_size_record(dependency)
    dependent = vpx.Candidate(
        "dependent",
        "dependent-row",
        {"axis": "dependent"},
        dependency_identities={
            "dependency": dependency_identity(dependency, dependency_record)
        },
        admission_status="passed",
    )
    dependent_record = full_size_record(dependent)
    plan = vpx.Plan(
        selected={"dependency": dependency, "dependent": dependent},
        records={
            "dependency": dependency_record,
            "dependent": dependent_record,
        },
        input_signature={"case": "pilot"},
        policy=vpx.SelectionPolicy(),
        full_size_records=(dependency_record, dependent_record),
        materializers={
            "dependency": materialize_candidate,
            "dependent": materialize_candidate,
        },
        validation_order=("dependency", "dependent"),
        dependencies_by_family={"dependent": ("dependency",)},
        validation_required=True,
        validator_identities={
            "dependency": {"validator": "test"},
            "dependent": {"validator": "test"},
        },
    )

    def validator(
        candidate: vpx.Candidate,
        record: vpx.FullSizeRecord,
        context: vpx.PlanValidationContext,
    ) -> vpx.ReferenceResult:
        assert candidate.family == record.family
        assert context.family == candidate.family

        return reference_passed()

    validation_records = vp.validate_plan(
        plan,
        {
            "dependency": validator,
            "dependent": validator,
        },
    )
    bad_dependent_record = refresh_check_record(
        dataclasses.replace(
            validation_records[1],
            dependency_identities={
                "dependency": {
                    **dict(dependent.dependency_identities["dependency"]),
                    "full_size_row": {
                        **dict(
                            dependent.dependency_identities["dependency"][
                                "full_size_row"
                            ]
                        ),
                        "candidate_id": "different",
                    },
                }
            },
        )
    )

    with pytest.raises(vp.MaterializationError, match="dependencies"):
        selected_settings(
            plan,
            ("dependent",),
            validation_records=(validation_records[0], bad_dependent_record),
        )
