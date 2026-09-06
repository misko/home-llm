from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_lab.telemetry import NvidiaTelemetrySampler, parse_nvidia_smi


def test_parse_nvidia_smi_csv_and_na_values():
    output = (
        "0, GPU-abcd, NVIDIA GeForce RTX 4090, 62, 94, 31, "
        "21344, 24564, 401.25, 450.00, 2715\n"
        "1, GPU-efgh, NVIDIA A10, N/A, 0, 0, 12, 23028, N/A, 150, 300\n"
    )

    samples = parse_nvidia_smi(output, timestamp="2026-01-01T00:00:00Z")

    assert len(samples) == 2
    first, second = samples
    assert first.available
    assert first.index == 0
    assert first.name == "NVIDIA GeForce RTX 4090"
    assert first.memory_used_mib == 21344
    assert first.power_draw_w == 401.25
    assert second.temperature_c is None
    assert second.power_draw_w is None


def test_sampler_is_robust_when_nvidia_smi_is_missing():
    def missing(_):
        raise FileNotFoundError("nvidia-smi")

    sampler = NvidiaTelemetrySampler(command_runner=missing)
    sample, = sampler.sample_once()

    assert not sample.available
    assert "unavailable" in sample.error


def test_sampler_records_nonzero_exit_without_raising():
    sampler = NvidiaTelemetrySampler(
        command_runner=lambda _: SimpleNamespace(
            returncode=9, stdout="", stderr="driver communication failed"
        )
    )

    sample, = sampler.sample_once()

    assert not sample.available
    assert "driver communication failed" in sample.error


@pytest.mark.asyncio
async def test_async_sampler_collects_at_least_one_observation():
    output = "0, GPU-a, RTX 4090, 40, 2, 1, 100, 24564, 20, 450, 210\n"
    sampler = NvidiaTelemetrySampler(
        interval_seconds=60,
        command_runner=lambda _: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    await sampler.start()
    await sampler.stop()

    assert sampler.samples
    assert sampler.samples[0].available
