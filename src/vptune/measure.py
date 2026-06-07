"""Candidate measurement."""

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import torch

from vptune.data import (
    PACKAGE_VERSION,
    Candidate,
    FullSizeRecord,
    Measurement,
    TimingPolicy,
)
from vptune.errors import MeasurementError
from vptune.tensor_tree import TensorTree, tree_detach, tree_leaves, tree_signature


class MemoryBackend(Protocol):
    """Memory sampling backend."""

    def identity(self) -> Mapping[str, Any]:
        """Return stable memory measurement identity."""

    def prepare(self) -> None:
        """Reset memory state before a call."""

    def sample(self) -> tuple[Measurement, ...]:
        """Return memory samples after a call."""

    def synchronize(self) -> None:
        """Synchronize measured work."""

    def cleanup(self) -> None:
        """Release unused cached memory."""


class CPUMemoryBackend:
    """Zero-valued memory backend for CPU tests and CPU candidates."""

    @staticmethod
    def identity() -> Mapping[str, Any]:
        """Return stable CPU memory measurement identity."""
        return {
            "backend_id": "vptune.cpu_memory",
            "backend_version": PACKAGE_VERSION,
            "devices": ("cpu",),
            "prepare": "none",
            "synchronize": "none",
            "cleanup": "none",
        }

    @staticmethod
    def prepare() -> None:
        """Prepare CPU memory state."""

    @staticmethod
    def sample() -> tuple[Measurement, ...]:
        """Return one CPU memory sample."""
        return (
            Measurement(
                elapsed_seconds=0.0,
                peak_allocated_mib=0.0,
                peak_reserved_mib=0.0,
                post_allocated_mib=0.0,
                post_reserved_mib=0.0,
                rank=0,
                device="cpu",
            ),
        )

    @staticmethod
    def synchronize() -> None:
        """Synchronize CPU work."""

    @staticmethod
    def cleanup() -> None:
        """Clean up CPU memory state."""


class CUDAMemoryBackend:
    """CUDA memory backend for one or more devices."""

    def __init__(self, devices: Sequence[str] | None = None) -> None:
        """Initialize backend for selected devices."""
        if devices is None:
            self.devices = tuple(
                f"cuda:{index}" for index in range(torch.cuda.device_count())
            )
        else:
            self.devices = tuple(devices)

    def identity(self) -> Mapping[str, Any]:
        """Return stable CUDA memory measurement identity."""
        return {
            "backend_id": "vptune.cuda_memory",
            "backend_version": PACKAGE_VERSION,
            "devices": self.devices,
            "prepare": "empty_cache_reset_peak_synchronize",
            "synchronize": "cuda_synchronize_all_devices",
            "cleanup": "empty_cache",
            "device_memory_used": hasattr(torch.cuda, "device_memory_used"),
        }

    def prepare(self) -> None:
        """Reset CUDA peak memory on every device."""
        torch.cuda.empty_cache()

        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

    def sample(self) -> tuple[Measurement, ...]:
        """Return CUDA memory samples."""
        samples = []
        scale = 1024.0 * 1024.0

        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                used = None

                if hasattr(torch.cuda, "device_memory_used"):
                    used = torch.cuda.device_memory_used() / scale

                samples.append(
                    Measurement(
                        elapsed_seconds=0.0,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / scale,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / scale,
                        post_allocated_mib=torch.cuda.memory_allocated() / scale,
                        post_reserved_mib=torch.cuda.memory_reserved() / scale,
                        device_memory_used_mib=used,
                        rank=0,
                        device=str(torch.device(device)),
                    )
                )

        return tuple(samples)

    def synchronize(self) -> None:
        """Synchronize every configured CUDA device."""
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()

    @staticmethod
    def cleanup() -> None:
        """Release unused CUDA allocator cache."""
        torch.cuda.empty_cache()


def default_memory_backend(devices: Sequence[str]) -> MemoryBackend:
    """Return a memory backend for devices."""
    if any(str(device).startswith("cuda") for device in devices):
        return CUDAMemoryBackend(devices)

    return CPUMemoryBackend()


class OperationMeasurementError(MeasurementError):
    """Operation failure with samples from the measured region."""

    def __init__(
        self,
        *,
        error_type: str,
        error: str,
        samples: tuple[Measurement, ...],
    ) -> None:
        super().__init__(error)
        self.error_type = error_type
        self.error = error
        self.samples = samples


def clear_parameter_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    """Clear gradients on parameters."""
    for parameter in parameters:
        parameter.grad = None


def _sample_with_elapsed(
    memory_backend: MemoryBackend,
    elapsed_seconds: float,
) -> tuple[Measurement, ...]:
    return tuple(
        Measurement(
            elapsed_seconds=elapsed_seconds,
            peak_allocated_mib=sample.peak_allocated_mib,
            peak_reserved_mib=sample.peak_reserved_mib,
            post_allocated_mib=sample.post_allocated_mib,
            post_reserved_mib=sample.post_reserved_mib,
            device_memory_used_mib=sample.device_memory_used_mib,
            rank=sample.rank,
            device=sample.device,
        )
        for sample in memory_backend.sample()
    )


def measure_once(
    operation: Callable[[], TensorTree],
    *,
    memory_backend: MemoryBackend,
    clock: Callable[[], float] = time.perf_counter,
    clear_gradients: Callable[[], None] | None = None,
) -> tuple[float, tuple[Measurement, ...], TensorTree]:
    """Measure one operation call.

    Returns:
        Elapsed seconds, memory samples, and detached output.

    Raises:
        OperationMeasurementError: If the operation fails after timing starts.
    """
    memory_backend.cleanup()

    if clear_gradients is not None:
        clear_gradients()

    memory_backend.prepare()
    start = clock()

    try:
        output = operation()
    except torch.cuda.OutOfMemoryError as error:
        memory_backend.synchronize()
        end = clock()
        samples = _sample_with_elapsed(memory_backend, end - start)
        raise OperationMeasurementError(
            error_type=type(error).__name__,
            error=str(error),
            samples=samples,
        ) from error
    except RuntimeError as error:
        memory_backend.synchronize()
        end = clock()
        samples = _sample_with_elapsed(memory_backend, end - start)
        raise OperationMeasurementError(
            error_type=type(error).__name__,
            error=str(error),
            samples=samples,
        ) from error
    else:
        memory_backend.synchronize()
        end = clock()
        memory_samples = _sample_with_elapsed(memory_backend, end - start)

        return end - start, memory_samples, tree_detach(output)

    finally:
        memory_backend.cleanup()


def measure_operation(
    operation: Callable[[], TensorTree],
    *,
    timing_policy: TimingPolicy,
    memory_backend: MemoryBackend,
    clock: Callable[[], float] = time.perf_counter,
    clear_gradients: Callable[[], None] | None = None,
) -> tuple[tuple[Measurement, ...], TensorTree, tuple[Measurement, ...]]:
    """Measure an operation using the package timing policy.

    Returns:
        Measured samples, detached output, and probe-call samples.

    Raises:
        RuntimeError: If the timing policy asks for no measured calls.
    """
    probe_elapsed, probe_memory, probe_output = measure_once(
        operation,
        memory_backend=memory_backend,
        clock=clock,
        clear_gradients=clear_gradients,
    )
    warmups, measured_calls = timing_policy.plan(probe_elapsed)

    if measured_calls <= 0:
        message = "timing policy produced no measured calls"
        raise RuntimeError(message)

    if warmups == 0 and measured_calls == 1:
        return probe_memory, probe_output, probe_memory

    for _ in range(warmups):
        measure_once(
            operation,
            memory_backend=memory_backend,
            clock=clock,
            clear_gradients=clear_gradients,
        )

    samples = []
    output = probe_output

    for _ in range(measured_calls):
        _, memory_samples, output = measure_once(
            operation,
            memory_backend=memory_backend,
            clock=clock,
            clear_gradients=clear_gradients,
        )
        samples.extend(memory_samples)

    return tuple(samples), output, probe_memory


def run_candidate(
    candidate: Candidate,
    input_signature: dict[str, Any],
    operation: Callable[[], TensorTree],
    *,
    timing_policy: TimingPolicy,
    memory_backend: MemoryBackend,
    clock: Callable[[], float] = time.perf_counter,
    clear_gradients: Callable[[], None] | None = None,
    reference_passed: bool = True,
    full_size_check: Callable[
        [TensorTree, tuple[Measurement, ...]],
        Mapping[str, Any],
    ]
    | None = None,
) -> FullSizeRecord:
    """Run and record one full-size candidate.

    Returns:
        Passed or failed full-size record.
    """
    compile_counter_before = _compile_counter(candidate)

    try:
        samples, output, probe_samples = measure_operation(
            operation,
            timing_policy=timing_policy,
            memory_backend=memory_backend,
            clock=clock,
            clear_gradients=clear_gradients,
        )
        _require_finite_output(output)
        selection_metadata = _selection_metadata(
            candidate,
            samples,
            probe_samples,
            compile_counter_before=compile_counter_before,
            compile_counter_after=_compile_counter(candidate),
        )

        if full_size_check is not None:
            selection_metadata = {
                **selection_metadata,
                **dict(full_size_check(output, samples)),
            }
    except OperationMeasurementError as error:
        return failed_record(
            candidate,
            input_signature,
            error_type=error.error_type,
            error=error.error,
            reference_passed=reference_passed,
            samples=error.samples,
        )
    except torch.cuda.OutOfMemoryError as error:
        return failed_record(
            candidate,
            input_signature,
            error_type=type(error).__name__,
            error=str(error),
            reference_passed=reference_passed,
        )
    except RuntimeError as error:
        return failed_record(
            candidate,
            input_signature,
            error_type=type(error).__name__,
            error=str(error),
            reference_passed=reference_passed,
        )

    return FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=dict(input_signature),
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        timing_samples=samples,
        memory_samples=samples,
        output_signature=tree_signature(output),
        selection_metadata=selection_metadata,
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        reference_passed=reference_passed,
    )


def _require_finite_output(output: TensorTree) -> None:
    for tensor in tree_leaves(output):
        if not torch.isfinite(tensor).all().item():
            message = "full-size output contains nonfinite values"
            raise RuntimeError(message)


def _selection_metadata(
    candidate: Candidate,
    samples: tuple[Measurement, ...],
    probe_samples: tuple[Measurement, ...],
    *,
    compile_counter_before: int | None,
    compile_counter_after: int | None,
) -> dict[str, Any]:
    if candidate.settings.get("compile.enabled") != "true":
        return {"timing_source": "eager_single_rank"}

    steady_elapsed = _median_elapsed(samples)
    compile_time = _compile_time_seconds(candidate, steady_elapsed, probe_samples)

    return {
        "timing_source": "compiled_single_rank",
        "steady_elapsed_seconds": steady_elapsed,
        "compile_time_seconds": compile_time,
        "recompile_count": _recompile_count(
            compile_counter_before,
            compile_counter_after,
        ),
        "compile_cache_state": candidate.settings.get("compile.cache_state"),
        "compile.compiled_autograd": candidate.settings.get(
            "compile.compiled_autograd"
        ),
        "compile.cuda_graphs": candidate.settings.get("compile.cuda_graphs"),
    }


def _compile_time_seconds(
    candidate: Candidate,
    steady_elapsed: float,
    probe_samples: tuple[Measurement, ...],
) -> float:
    if candidate.settings.get("compile.cache_state") != "cold_compile":
        return 0.0

    probe_elapsed = _median_elapsed(probe_samples)
    compile_time = probe_elapsed - steady_elapsed

    if compile_time <= 0.0:
        return 0.0

    return compile_time


def _compile_counter(candidate: Candidate) -> int | None:
    if candidate.settings.get("compile.enabled") != "true":
        return None

    dynamo_utils = importlib.import_module("torch._dynamo.utils")

    return int(dynamo_utils.counters["stats"]["unique_graphs"])


def _recompile_count(before: int | None, after: int | None) -> int:
    if before is None or after is None:
        return 0

    graph_delta = after - before

    if graph_delta <= 1:
        return 0

    return graph_delta - 1


def _median_elapsed(samples: tuple[Measurement, ...]) -> float:
    values = sorted(sample.elapsed_seconds for sample in samples)

    if not values:
        message = "selection metadata has no timing samples"
        raise RuntimeError(message)

    midpoint = len(values) // 2

    if len(values) % 2 == 1:
        return values[midpoint]

    return 0.5 * (values[midpoint - 1] + values[midpoint])


def failed_record(
    candidate: Candidate,
    input_signature: dict[str, Any],
    *,
    error_type: str,
    error: str,
    reference_passed: bool,
    samples: tuple[Measurement, ...] = (),
) -> FullSizeRecord:
    """Return a failed full-size record."""
    return FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="failed",
        input_signature=dict(input_signature),
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        reference_passed=reference_passed,
        error_type=error_type,
        error=error,
        timing_samples=samples,
        memory_samples=samples,
    )
