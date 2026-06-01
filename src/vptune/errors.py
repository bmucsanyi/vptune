"""Package error hierarchy."""


class VPTuneError(RuntimeError):
    """Base class for package errors."""


class AdmissionError(VPTuneError):
    """Raised when a candidate cannot preserve declared semantics."""


class ReferenceFailedError(VPTuneError):
    """Raised when a candidate fails a reference check."""


class NoPassedCandidateError(VPTuneError):
    """Raised when selection has no accepted candidate."""


class StaleRecordError(VPTuneError):
    """Raised when a saved record does not match the current run."""


class MeasurementError(VPTuneError):
    """Raised when candidate measurement cannot produce valid samples."""


class MaterializationError(VPTuneError):
    """Raised when a selected plan cannot be materialized."""
