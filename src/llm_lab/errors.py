"""Domain-specific exceptions used throughout LLM Lab."""


class LabError(RuntimeError):
    """Base error for an expected LLM Lab failure."""


class CatalogError(LabError):
    """A catalog entry is missing, conflicting, or invalid."""


class IntegrityError(LabError):
    """Artifact bytes do not match their immutable manifest."""


class StoragePolicyError(LabError):
    """An operation would violate the configured storage policy."""


class DeploymentError(LabError):
    """A model deployment could not be activated or reached."""


class BenchmarkError(LabError):
    """A benchmark run is invalid or could not complete."""
