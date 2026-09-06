"""Best-effort NVIDIA GPU telemetry collection.

Telemetry must never be the reason a benchmark fails.  Missing drivers, a
machine without a GPU, transient ``nvidia-smi`` failures, and individual
``N/A`` fields are represented explicitly instead of raising from the sampler.
"""

from __future__ import annotations

import asyncio
import csv
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from typing import Any


NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.sm",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _optional_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.casefold() in {"n/a", "na", "not supported", "[n/a]"}:
        return None
    # nounits is requested, but accepting common suffixes makes fixtures and
    # output from older nvidia-smi releases parse safely.
    for suffix in (" MiB", " W", " %", " MHz", " C"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _optional_int(value: str) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One GPU observation, or an explicit unavailable observation."""

    timestamp: str
    available: bool
    index: int | None = None
    uuid: str | None = None
    name: str | None = None
    temperature_c: float | None = None
    gpu_utilization_percent: float | None = None
    memory_utilization_percent: float | None = None
    memory_used_mib: float | None = None
    memory_total_mib: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    sm_clock_mhz: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable_sample(error: str, *, timestamp: str | None = None) -> TelemetrySample:
    return TelemetrySample(timestamp=timestamp or _utc_now(), available=False, error=error)


def parse_nvidia_smi(
    output: str,
    *,
    timestamp: str | None = None,
) -> tuple[TelemetrySample, ...]:
    """Parse CSV output generated for :data:`NVIDIA_SMI_FIELDS`.

    Malformed rows become unavailable observations.  This preserves the
    sampling timeline while allowing well-formed rows from other GPUs through.
    """

    observed_at = timestamp or _utc_now()
    if not output.strip():
        return (unavailable_sample("nvidia-smi returned no GPU rows", timestamp=observed_at),)

    parsed: list[TelemetrySample] = []
    reader = csv.reader(StringIO(output))
    for row_number, row in enumerate(reader, start=1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(NVIDIA_SMI_FIELDS):
            parsed.append(unavailable_sample(
                f"malformed nvidia-smi row {row_number}: expected "
                f"{len(NVIDIA_SMI_FIELDS)} fields, got {len(row)}",
                timestamp=observed_at,
            ))
            continue
        values = [cell.strip() for cell in row]
        index = _optional_int(values[0])
        if index is None:
            parsed.append(unavailable_sample(
                f"malformed nvidia-smi row {row_number}: invalid GPU index",
                timestamp=observed_at,
            ))
            continue
        parsed.append(TelemetrySample(
            timestamp=observed_at,
            available=True,
            index=index,
            uuid=values[1] or None,
            name=values[2] or None,
            temperature_c=_optional_float(values[3]),
            gpu_utilization_percent=_optional_float(values[4]),
            memory_utilization_percent=_optional_float(values[5]),
            memory_used_mib=_optional_float(values[6]),
            memory_total_mib=_optional_float(values[7]),
            power_draw_w=_optional_float(values[8]),
            power_limit_w=_optional_float(values[9]),
            sm_clock_mhz=_optional_float(values[10]),
        ))
    return tuple(parsed) or (
        unavailable_sample("nvidia-smi returned no GPU rows", timestamp=observed_at),
    )


CommandRunner = Callable[[Sequence[str]], Any]


def _default_command_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )


class NvidiaTelemetrySampler:
    """Periodically sample NVIDIA telemetry without failing its caller."""

    def __init__(
        self,
        interval_seconds: float = 1.0,
        *,
        command_runner: CommandRunner | None = None,
        executable: str = "nvidia-smi",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.command_runner = command_runner or _default_command_runner
        self.executable = executable
        self.samples: list[TelemetrySample] = []
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def command(self) -> tuple[str, ...]:
        fields = ",".join(NVIDIA_SMI_FIELDS)
        return (
            self.executable,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        )

    def sample_once(self) -> tuple[TelemetrySample, ...]:
        """Collect one timestamp, returning an unavailable row on any failure."""

        observed_at = _utc_now()
        try:
            result = self.command_runner(self.command)
            if isinstance(result, str):
                stdout, stderr, returncode = result, "", 0
            elif isinstance(result, tuple):
                stdout = str(result[0]) if result else ""
                stderr = str(result[1]) if len(result) > 1 else ""
                returncode = int(result[2]) if len(result) > 2 else 0
            else:
                stdout = str(getattr(result, "stdout", "") or "")
                stderr = str(getattr(result, "stderr", "") or "")
                returncode = int(getattr(result, "returncode", 0))
            if returncode != 0:
                detail = stderr.strip() or f"exit status {returncode}"
                return (unavailable_sample(f"nvidia-smi failed: {detail}", timestamp=observed_at),)
            return parse_nvidia_smi(stdout, timestamp=observed_at)
        except FileNotFoundError:
            return (unavailable_sample("nvidia-smi is unavailable", timestamp=observed_at),)
        except subprocess.TimeoutExpired:
            return (unavailable_sample("nvidia-smi timed out", timestamp=observed_at),)
        except Exception as exc:
            return (unavailable_sample(
                f"nvidia-smi collection failed: {type(exc).__name__}: {exc}",
                timestamp=observed_at,
            ),)

    async def _sample_once_async(self) -> tuple[TelemetrySample, ...]:
        return await asyncio.to_thread(self.sample_once)

    async def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            self.samples.extend(await self._sample_once_async())
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue

    async def start(self) -> None:
        """Start sampling immediately; repeated calls while running are harmless."""

        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="nvidia-telemetry")
        # Yield so even extremely short benchmarks normally retain one sample.
        await asyncio.sleep(0)

    async def stop(self) -> tuple[TelemetrySample, ...]:
        """Stop sampling and return all observations collected so far."""

        if self._task is None:
            return tuple(self.samples)
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop_event = None
        return tuple(self.samples)

    async def __aenter__(self) -> "NvidiaTelemetrySampler":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()


# A shorter spelling for callers that already know the backend.
NvidiaSampler = NvidiaTelemetrySampler
NvidiaSmiSampler = NvidiaTelemetrySampler
parse_nvidia_smi_csv = parse_nvidia_smi
