import warnings

import pytest

from rembg.sessions.base import BaseSession


class FakeInner:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


def make_session(active):
    """A BaseSession whose session reports `active` as its providers.

    Built without __init__ so the check can be exercised without onnxruntime
    creating a real session or a model being downloaded.
    """
    session = BaseSession.__new__(BaseSession)
    session.inner_session = FakeInner(active)
    return session


def test_warns_when_cuda_requested_but_cpu_active():
    """The silent CPU fallback reported in #841 now surfaces as a warning."""
    session = make_session(["CPUExecutionProvider"])

    with pytest.warns(RuntimeWarning, match="running on CPU"):
        session._warn_on_provider_fallback(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )


def test_no_warning_when_cuda_is_active():
    session = make_session(["CUDAExecutionProvider", "CPUExecutionProvider"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session._warn_on_provider_fallback(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )


def test_no_warning_when_cpu_was_what_was_asked_for():
    """A CPU-only install is not a fallback and must stay quiet."""
    session = make_session(["CPUExecutionProvider"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session._warn_on_provider_fallback(["CPUExecutionProvider"])


@pytest.mark.parametrize(
    "provider",
    ["ROCMExecutionProvider", "OpenVINOExecutionProvider"],
)
def test_warns_for_other_accelerators(provider):
    """The check is not CUDA-specific."""
    session = make_session(["CPUExecutionProvider"])

    with pytest.warns(RuntimeWarning, match=provider):
        session._warn_on_provider_fallback([provider, "CPUExecutionProvider"])


def test_partial_fallback_does_not_warn():
    """If any accelerator survived, the session is not on CPU alone."""
    session = make_session(["ROCMExecutionProvider", "CPUExecutionProvider"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session._warn_on_provider_fallback(
            ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        )


def test_survives_a_session_that_cannot_report_providers():
    """A backend without get_providers must not break session creation."""

    class Broken:
        def get_providers(self):
            raise RuntimeError("nope")

    session = BaseSession.__new__(BaseSession)
    session.inner_session = Broken()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session._warn_on_provider_fallback(["CUDAExecutionProvider"])
