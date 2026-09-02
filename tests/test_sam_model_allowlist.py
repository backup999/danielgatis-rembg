import pytest

from rembg.sessions.sam import SAM_MODELS, SamSession


@pytest.mark.parametrize(
    "sam_model",
    [
        "../../../../etc/rembg_evil",
        "sam_vit_b_01ec64/../../../evil",
        "sam_vit_b_01ec64#/../../evil",
        "/etc/passwd",
        "sam_vit_b_01ec64\x00evil",
        "",
    ],
)
def test_download_models_rejects_injected_names(sam_model, tmp_path, monkeypatch):
    """A `sam_model` that could steer a path or a URL never reaches pooch.

    `sam_model` arrives straight from the public `extras` payload of
    `/api/remove`, so a traversal or fragment in it must be refused before it is
    interpolated into a download URL or a local filename.
    """
    monkeypatch.setenv("REMBG_HOME", str(tmp_path))

    with pytest.raises(ValueError, match="unknown sam_model"):
        SamSession.download_models(sam_model=sam_model)


def test_download_models_rejects_before_any_download(tmp_path, monkeypatch):
    """Rejection happens before pooch is called at all, not after a fetch."""
    monkeypatch.setenv("REMBG_HOME", str(tmp_path))

    def explode(*args, **kwargs):
        raise AssertionError("pooch.retrieve must not run for a rejected sam_model")

    monkeypatch.setattr("rembg.sessions.sam.pooch.retrieve", explode)

    with pytest.raises(ValueError, match="unknown sam_model"):
        SamSession.download_models(sam_model="../../evil")


def test_allowlist_holds_only_published_checkpoints():
    """The allowlist tracks the SAM checkpoints published as release assets."""
    assert SAM_MODELS == {
        "sam_vit_b_01ec64",
        "sam_vit_l_0b3195",
        "sam_vit_h_4b8939",
    }


def test_injected_name_never_reaches_a_url_or_path(tmp_path, monkeypatch):
    """The guard is what stops injection, not the shape of the call below it.

    Records every URL and path pooch would be handed. Before the allowlist this
    captured a traversal in `fname` and a `#fragment` in the URL; now nothing is
    recorded because the call is refused first.
    """
    monkeypatch.setenv("REMBG_HOME", str(tmp_path))

    seen = []

    def record(url, known_hash, **kwargs):
        seen.append((url, kwargs.get("fname"), kwargs.get("path")))
        raise AssertionError(f"pooch.retrieve reached with {url!r}")

    monkeypatch.setattr("rembg.sessions.sam.pooch.retrieve", record)

    with pytest.raises(ValueError, match="unknown sam_model"):
        SamSession.download_models(sam_model="sam_vit_b_01ec64#/../../../evil")

    assert seen == []


def test_every_allowed_model_has_pinned_checksums():
    """Each allowed checkpoint has a digest for all four of its assets.

    SAM previously downloaded with no hash at all. If a new checkpoint is added
    to the allowlist without its digests, it would silently download unpinned.
    """
    from rembg.sessions.sam import SAM_CHECKSUMS

    for model in SAM_MODELS:
        for part in ("encoder", "decoder"):
            for suffix in (".onnx", ".quant.onnx"):
                fname = f"{model}.{part}{suffix}"
                assert fname in SAM_CHECKSUMS, f"missing checksum for {fname}"
                assert SAM_CHECKSUMS[fname].startswith("md5:")


def test_vit_h_sidecar_blobs_are_pinned():
    """The large encoder's three sidecar blobs are pinned too."""
    from rembg.sessions.sam import SAM_CHECKSUMS

    for i in (1, 2, 3):
        fname = f"sam_vit_h_4b8939.encoder_data.{i}.bin"
        assert fname in SAM_CHECKSUMS, f"missing checksum for {fname}"


def test_checksums_cover_only_allowed_models():
    """No digest refers to a checkpoint outside the allowlist."""
    from rembg.sessions.sam import SAM_CHECKSUMS

    for fname in SAM_CHECKSUMS:
        assert any(fname.startswith(m + ".") for m in SAM_MODELS), fname
