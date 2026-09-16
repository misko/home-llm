"""Coordinate a training process with existing serving/benchmark transitions."""
from contextlib import contextmanager
from collections.abc import Iterator

from .errors import DeploymentError
from .paths import LabPaths


def assert_training_idle(paths: LabPaths) -> None:
    """Called under gpu0.lock; the kernel lock, not stale JSON, owns the GPU."""
    from .runtime import ExclusiveGpuLock
    try:
        with ExclusiveGpuLock(paths.data_root / "state/training.lock", timeout_seconds=0):
            pass
    except DeploymentError as exc:
        raise DeploymentError("GPU is reserved by a training job; inspect its training run status") from exc


@contextmanager
def training_lease(paths: LabPaths) -> Iterator[None]:
    """The actual training process holds this lease for its complete lifetime."""
    from .runtime import ExclusiveGpuLock, read_active_state
    paths.initialize()
    lease = ExclusiveGpuLock(paths.data_root / "state/training.lock", timeout_seconds=0)
    with ExclusiveGpuLock(paths.gpu_lock_path):
        if read_active_state(paths) is not None:
            raise DeploymentError("Stop the active LLM Lab deployment before training")
        lease.__enter__()
    try:
        yield
    finally:
        lease.__exit__(None, None, None)
