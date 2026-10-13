"""Apply worker compile-cache settings after JAX's import-time env snapshot."""


def configure_compile_cache(runtime_env):
    import jax

    expected = {}
    for env, option, convert in (
        ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "jax_persistent_cache_min_compile_time_secs", float),
        ("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "jax_persistent_cache_min_entry_size_bytes", int),
        ("JAX_LOG_COMPILES", "jax_log_compiles", _boolean),
        ("JAX_EXPLAIN_CACHE_MISSES", "jax_explain_cache_misses", _boolean),
    ):
        if env in runtime_env:
            expected[option] = convert(runtime_env[env])
    for option, value in expected.items():
        jax.config.update(option, value)
    verify_compile_cache(expected)
    return expected


def verify_compile_cache(expected):
    import jax

    for option, requested in expected.items():
        effective = getattr(jax.config, option)
        if effective != requested:
            raise RuntimeError(
                f"Effective JAX compile-cache setting {option}={effective!r} "
                f"differs from requested {requested!r}")


def _boolean(value):
    normalized = value.lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid JAX boolean setting: {value!r}")
