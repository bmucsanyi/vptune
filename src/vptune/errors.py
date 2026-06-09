"""Package error hierarchy."""


class VPTuneError(RuntimeError):
    """Base class for package errors."""


class AdmissionError(VPTuneError):
    """Raised when a candidate cannot preserve declared semantics."""


class ReferenceFailedError(VPTuneError):
    """Raised when a candidate fails a reference check."""


class RecordFormatError(VPTuneError):
    """Raised when a saved record is malformed."""


class NoPassedCandidateError(VPTuneError):
    """Raised when selection has no accepted candidate."""


class StaleRecordError(VPTuneError):
    """Raised when a saved record does not match the current run."""


class MeasurementError(VPTuneError):
    """Raised when measurement cannot produce a usable row."""


class MaterializationError(VPTuneError):
    """Raised when a selected plan cannot be materialized."""


class CompileSetupError(MaterializationError):
    """Raised when declared compile settings cannot be set up."""


class RecordValidationError(MaterializationError):
    """Raised when declared record or data content is invalid."""


class RuntimeValueError(MaterializationError):
    """Raised when a declared runtime value cannot be converted or validated."""
