"""MLX memory caps for the 16 GB Mac mini (one MLX job at a time; see results/llmdec/status_incident.json)."""

from __future__ import annotations

import os


def cap(mem_gb: float | None = None, cache_gb: float = 1.0) -> None:
    import mlx.core as mx
    mem_gb = float(os.environ.get("LLMDEC_MEM_GB", mem_gb or 9.5))
    for name, val in (("set_memory_limit", mem_gb), ("set_cache_limit", cache_gb)):
        fn = getattr(mx, name, None) or getattr(mx.metal, name)
        fn(int(val * 1024 ** 3))
    print(f"[mlxsafe] memory limit {mem_gb} GB, cache limit {cache_gb} GB", flush=True)


def peak_gb() -> float:
    import mlx.core as mx
    fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return fn() / 1024 ** 3
