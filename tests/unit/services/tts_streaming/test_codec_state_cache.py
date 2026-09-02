"""Tests for codec state caching across chunks — Story 20.5 Phase 2 (AC #3).

WHY THIS FILE CAN ASSERT EXACTNESS RATHER THAN STATISTICS
---------------------------------------------------------
Most changes to a decode path can only be regression-tested against a
tolerance. This one cannot be *wrong* by a little: Story 20.5 Phase 1
(evidence §2.5) established that a streaming decode with state carried is

  * **bit-for-bit identical** to ``Qwen3TTSTokenizerV2Decoder.forward`` when
    the whole sequence is handed to it in one call, and
  * identical to the whole-sequence decode to the float floor (7.7e-07 in
    fp32 with TF32 off) when it is chunked.

So that is the bar this file holds the implementation to, and the tests run
on a **real ``Qwen3TTSTokenizerV2Decoder``** — the genuine upstream classes,
built tiny (2 transformer layers, latent 8, 8 samples/frame) with randomised
weights so it runs on CPU in float64 in CI. Nothing here is a mock of the
decoder: a mocked decoder could not detect the one bug that matters.

THE BUG THAT MATTERS
--------------------
``nn.ConvTranspose1d`` adds its bias to every output position of *each*
partial convolution, so a naive overlap-add double-counts it. Uncorrected it
leaves a bias-shaped transient at every seam that survives an fp32 pass and
reads as a ~17-21 % residual cold start — that is, it looks exactly like the
Phase 1 NO-GO verdict rather than like a bug, and it cost the Phase 1 spike
two runs before ``20-5-stage-probe.py`` localised it.
``test_transposed_conv_bias_is_not_double_counted`` pins it directly, by
running the naive form and requiring it to be *orders of magnitude* worse.
Without that test a regression here would be reported as "state caching does
not work".
"""

import numpy as np
import pytest
import torch

from myvoice.services.tts_streaming import codec_state_cache as csc
from myvoice.services.tts_streaming.streaming_decoder import (
    _CODEC_EDGE_LOSS_SAMPLES,
    _CODEC_SAMPLES_PER_FRAME,
)


# --------------------------------------------------------------------------- #
# A real (tiny) decoder
# --------------------------------------------------------------------------- #


def _build_tiny_decoder(dtype=torch.float64, **overrides):
    """A genuine ``Qwen3TTSTokenizerV2Decoder``, small enough for CPU CI.

    Geometry mirrors the shipping model's *structure* exactly — a stride-1
    causal ``pre_conv``, an all-sliding transformer, ``upsampling_ratios``
    transposed convs with ``kernel == stride`` (stateless), and
    ``upsample_rates`` decoder blocks with ``kernel == 2*stride`` (the ones
    that own the edge loss) — at 8 samples/frame instead of 1920.

    ``post_init`` leaves the quantizer codebook and every bias at zero, which
    would make the whole decoder output silence and every assertion below
    vacuously true. So the weights are randomised explicitly. The nonzero
    ``ConvTranspose1d`` bias is what gives
    ``test_transposed_conv_bias_is_not_double_counted`` something to detect.
    """
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    kwargs = dict(
        codebook_size=16, codebook_dim=8, hidden_size=16, latent_dim=8,
        num_attention_heads=2, num_key_value_heads=2, intermediate_size=16,
        num_hidden_layers=2, num_quantizers=2, sliding_window=4,
        upsample_rates=(2, 2), upsampling_ratios=(2,), decoder_dim=8,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    decoder = Qwen3TTSTokenizerV2Decoder(
        Qwen3TTSTokenizerV2DecoderConfig(**kwargs)
    ).eval()
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, param in decoder.named_parameters():
            param.copy_(torch.empty(param.shape).uniform_(-0.3, 0.3, generator=gen))
            if "cluster_usage" in name:
                # Divided by in EuclideanCodebook.decode; keep it away from 0.
                param.copy_(param.abs() + 1.0)
    return decoder.to(dtype)


@pytest.fixture(scope="module")
def tiny_decoder():
    return _build_tiny_decoder()


@pytest.fixture(scope="module")
def tiny_geometry(tiny_decoder):
    return csc.probe_decoder(tiny_decoder)


def _codes(n_frames, num_quantizers=2, codebook_size=16, seed=3):
    return torch.randint(
        1, codebook_size, (1, num_quantizers, n_frames),
        generator=torch.Generator().manual_seed(seed),
    )


def _stream_chunked(decoder, codes, chunk, state=None):
    state = state if state is not None else csc.CodecStreamState()
    parts = []
    with torch.inference_mode():
        for i in range(0, codes.shape[-1], chunk):
            parts.append(csc.stream_forward(decoder, codes[..., i:i + chunk], state))
    return torch.cat(parts, dim=-1), state


def _nrmse(reference, test):
    reference = reference.to(torch.float64)
    test = test.to(torch.float64)
    denom = torch.linalg.vector_norm(reference).item()
    assert denom > 0, "reference signal is silent — the test would be vacuous"
    return torch.linalg.vector_norm(reference - test).item() / denom


# ============================================================================
# The regression bar (Phase 1 §2.5) — exact, not statistical
# ============================================================================


def test_single_chunk_streaming_is_bit_for_bit_identical_to_forward(tiny_decoder):
    """Bar 1: hand the whole sequence to the streaming traversal in ONE call
    and it must equal ``decoder.forward`` exactly.

    This is the strongest available statement that the traversal restates
    upstream's ``forward`` faithfully rather than approximately. It isolates
    the traversal from the state carrying: with one call there is no
    boundary, so any difference at all is a transcription error in the module
    walk — a wrong ordering, a missed residual, a dropped permute.
    """
    codes = _codes(40)
    with torch.inference_mode():
        reference = tiny_decoder(codes)
        streamed = csc.stream_forward(tiny_decoder, codes, csc.CodecStreamState())

    assert streamed.shape == reference.shape
    assert torch.equal(streamed, reference), (
        "single-chunk streaming diverged from decoder.forward. The module "
        "traversal in codec_state_cache.stream_forward no longer matches "
        "Qwen3TTSTokenizerV2Decoder.forward — compare them line by line "
        "before touching anything else."
    )


@pytest.mark.parametrize("chunk", [5, 8, 10, 13])
def test_chunked_with_state_equals_whole_sequence_to_the_float_floor(
    tiny_decoder, chunk
):
    """Bar 2: chunked-with-state == whole-sequence decode, to ~1e-6.

    Run in float64 so the assertion has room: Phase 1 measured 7.7e-07 in
    fp32 with TF32 off, and the tolerance below is two orders tighter than
    that because the arithmetic here is wider. The four chunk sizes include
    ones that do NOT divide the sequence evenly, so the residual-length path
    is covered too.
    """
    codes = _codes(40)
    with torch.inference_mode():
        whole = tiny_decoder(codes)
    streamed, _ = _stream_chunked(tiny_decoder, codes, chunk)

    assert streamed.shape == whole.shape, (
        f"chunked decode returned {tuple(streamed.shape)} samples, "
        f"whole-sequence returned {tuple(whole.shape)} — the edge loss has "
        f"stopped moving to the stream-start call."
    )
    assert _nrmse(whole, streamed) < 1e-9


def test_transposed_conv_bias_is_not_double_counted(tiny_decoder, monkeypatch):
    """THE trap. ``nn.ConvTranspose1d`` adds its bias to every output
    position of each partial convolution, so a naive overlap-add counts it
    twice at every seam.

    Uncorrected this leaves a bias-shaped transient that decays over ~2,000
    samples on the real model, survives an fp32 pass, and reads as a
    17-21 % residual cold start — i.e. it presents as "codec state caching
    does not remove the defect" rather than as a bug. That is why this test
    exists as its own row: the generic exactness tests above would fail too,
    but they would not *say* what broke.

    The test runs the naive form deliberately and requires it to be orders of
    magnitude worse than the corrected one, so it fails loudly in the
    direction of the real defect rather than on a tolerance nudge.
    """
    codes = _codes(40)
    with torch.inference_mode():
        whole = tiny_decoder(codes)

    corrected, _ = _stream_chunked(tiny_decoder, codes, 10)
    corrected_nrmse = _nrmse(whole, corrected)

    def _naive_tconv(module, x, state, key):
        """``_stream_tconv`` with the one-line bias correction removed."""
        left_pad, right_pad = module.left_pad, module.right_pad
        if left_pad == 0 and right_pad == 0:
            return module.conv(x).contiguous()
        y = module.conv(x)
        prev = state.tconv.get(key)
        if prev is not None:
            y = torch.cat([y[..., :left_pad] + prev, y[..., left_pad:]], dim=-1)
        state.tconv[key] = y[..., y.shape[-1] - right_pad:].clone()
        out = y[..., : y.shape[-1] - right_pad]
        if prev is None:
            out = out[..., left_pad:]
        return out.contiguous()

    monkeypatch.setattr(csc, "_stream_tconv", _naive_tconv)
    naive, _ = _stream_chunked(tiny_decoder, codes, 10)
    naive_nrmse = _nrmse(whole, naive)

    # The naive form gets the LENGTH right — which is exactly why it is
    # dangerous: every geometry check still passes.
    assert naive.shape == whole.shape
    assert naive_nrmse > 1e-3, (
        "the naive overlap-add (bias double-counted) scored "
        f"{naive_nrmse:.3e}, which is not distinguishable from the corrected "
        "form. Either the model under test has zero ConvTranspose1d bias — "
        "making this test vacuous — or the correction moved somewhere this "
        "monkeypatch no longer reaches."
    )
    assert naive_nrmse > corrected_nrmse * 1e6, (
        f"corrected={corrected_nrmse:.3e} naive={naive_nrmse:.3e}: the bias "
        "correction in _stream_tconv is not doing the work it is supposed to."
    )


def test_carried_state_removes_the_edge_loss_that_independent_decodes_pay(
    tiny_decoder, tiny_geometry
):
    """The Phase 1 headline, asserted structurally.

    Independently-decoded chunks each pay the fixed edge loss, so their
    concatenation is short by ``edge_loss`` per seam. With state carried the
    loss is paid once, at the stream-start call, exactly as the whole-
    sequence decode pays it — so the totals match to the sample.
    """
    codes = _codes(40)
    chunk = 10
    with torch.inference_mode():
        whole = tiny_decoder(codes)
        independent = torch.cat(
            [tiny_decoder(codes[..., i:i + chunk]) for i in range(0, 40, chunk)],
            dim=-1,
        )
    streamed, _ = _stream_chunked(tiny_decoder, codes, chunk)

    seams = 40 // chunk - 1
    assert whole.shape[-1] - independent.shape[-1] == seams * tiny_geometry.edge_loss_samples
    assert streamed.shape[-1] == whole.shape[-1]


def test_independent_decodes_reproduce_the_story_20_4_cold_start_defect(
    tiny_decoder
):
    """A control: the defect this story removes must be *present* on the
    stateless arm of the very same model, or the tests above prove nothing.

    Story 20.4 measured ~35 % NRMSE at the chunk head between two decodes of
    the same instant. The tiny model reproduces the same order of magnitude,
    which is what makes it a valid stand-in.
    """
    codes = _codes(40)
    chunk = 10
    with torch.inference_mode():
        whole = tiny_decoder(codes)
        second_chunk = tiny_decoder(codes[..., chunk:2 * chunk])

    spf = 8
    head = 32
    start = chunk * spf
    cold = _nrmse(whole[0, 0, start:start + head], second_chunk[0, 0, :head])
    assert cold > 0.1, (
        f"the stateless chunk head is only {cold:.3f} off the ground truth; "
        "the fixture no longer reproduces the cold-start defect and the "
        "exactness assertions above have nothing to distinguish themselves "
        "from."
    )


# ============================================================================
# AC #3 — per-session state, and the three reset points
# ============================================================================


def test_state_is_per_instance_not_module_or_class_level(tiny_decoder, tiny_geometry):
    """Concurrent generations are reachable through the HTTP API, so two
    decoders must not be able to see each other's codec state.

    Driven differently on purpose: if any state were class- or module-level,
    the second decoder's output would be contaminated by the first's frames.
    """
    left = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    right = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)

    codes_a = _codes(13, seed=3)[0].transpose(0, 1)
    codes_b = _codes(13, seed=99)[0].transpose(0, 1)

    left(codes_a)
    left(codes_a)
    solo = right(codes_b)

    fresh = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    np.testing.assert_array_equal(solo, fresh(codes_b))

    assert left.state_frames == 20
    assert right.state_frames == 10
    assert left._state is not right._state
    assert left._state.conv is not right._state.conv


def test_reset_returns_the_decoder_to_a_fresh_stream(tiny_decoder, tiny_geometry):
    """Reset must be a true reset, not a partial teardown: the first decode
    after it has to be byte-identical to the first decode of a brand-new
    decoder, including the 555-sample edge loss it pays again.
    """
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    chunk = _codes(13)[0].transpose(0, 1)

    first = decoder(chunk)
    decoder(chunk)
    assert decoder.state_frames == 20

    decoder.reset()
    assert decoder.state_frames == 0
    assert decoder._state.conv == {}
    assert decoder._state.tconv == {}
    assert decoder._state.kv is None

    np.testing.assert_array_equal(decoder(chunk), first)


def test_reset_is_idempotent_and_safe_before_any_decode(tiny_decoder, tiny_geometry):
    """The worker calls reset on the cancel path, where it may never have
    decoded anything and may never raise."""
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    decoder.reset()
    decoder.reset()
    assert decoder.state_frames == 0


def test_state_size_is_bounded_and_does_not_grow_with_utterance_length(
    tiny_decoder, tiny_geometry
):
    """AC #2's cost claim, held as a *property* rather than a number.

    The KV cache self-bounds because every layer is ``sliding_attention``;
    the conv/tconv buffers are fixed-depth by construction. So a long
    utterance must not cost more carried state than a short one.
    """
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    chunk = _codes(13)[0].transpose(0, 1)

    decoder(chunk)
    decoder(chunk)
    early_bytes, early_tensors = decoder.state_nbytes()
    for _ in range(20):
        decoder(chunk)
    late_bytes, late_tensors = decoder.state_nbytes()

    assert early_tensors == late_tensors
    assert late_bytes == early_bytes, (
        f"carried state grew from {early_bytes} to {late_bytes} bytes over "
        "20 further chunks. The KV cache is no longer bounded by "
        "sliding_window and the AC #2 per-session cost claim is void."
    )


# ============================================================================
# Geometry — the identities the worker's splice arithmetic depends on
# ============================================================================


def test_first_decode_pays_the_edge_loss_and_every_later_one_does_not(
    tiny_decoder, tiny_geometry
):
    """``1920*N - 555`` once, then exactly ``1920*N``. The 555 does not
    shrink — it moves to the stream-start call, which is where the whole-
    sequence decode loses it too."""
    spf = tiny_geometry.samples_per_frame
    edge = tiny_geometry.edge_loss_samples
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    chunk = _codes(13)[0].transpose(0, 1)

    assert decoder(chunk).size == spf * 13 - edge
    assert decoder(chunk).size == spf * 13
    assert decoder(chunk).size == spf * 13


def test_geometry_is_derived_from_the_loaded_modules_not_assumed(tiny_geometry):
    """``probe_decoder`` simulates the length arithmetic over the real
    modules. On the tiny fixture that is 2*2*2 = 8 samples/frame with a
    2*2 + 2 = 6-sample edge loss — the same formula that yields 1920 and 555
    on the shipping model."""
    assert tiny_geometry.samples_per_frame == 8
    assert tiny_geometry.edge_loss_samples == 6
    assert tiny_geometry.kv_layers == 2
    assert tiny_geometry.output_samples(13, first_call=True) == 8 * 13 - 6
    assert tiny_geometry.output_samples(13, first_call=False) == 8 * 13


def test_shipping_constants_are_what_the_worker_pins():
    """The two numbers ``streaming_decoder`` measured, restated here so a
    change to either fails in both files."""
    assert _CODEC_SAMPLES_PER_FRAME == 1920
    assert _CODEC_EDGE_LOSS_SAMPLES == 555
    # 8*(5*4*3) + 5*(4*3) + 4*3 + 3 — the transposed-conv decomposition
    # Phase 1 attributed 100 % of the edge loss to.
    assert 8 * 60 + 5 * 12 + 4 * 3 + 3 == _CODEC_EDGE_LOSS_SAMPLES
    assert 8 * 5 * 4 * 3 * 2 * 2 == _CODEC_SAMPLES_PER_FRAME


def test_lookahead_frames_are_decoded_but_not_committed(tiny_decoder, tiny_geometry):
    """The commit point is the SPLICE, not the end of the window.

    The streamer emits ``chunk_size + lookahead`` frames and slides by
    ``chunk_size``, so committing the whole window would leave the next chunk
    resuming ``lookahead`` frames in its own future — which is the one way
    state carrying can be wired up backwards and still produce plausible-
    looking audio.
    """
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    chunk = _codes(13)[0].transpose(0, 1)
    decoder(chunk)
    assert decoder.state_frames == 10, (
        "the decoder committed the lookahead frames to state; the next chunk "
        "would resume in the future and every seam would skip audio."
    )


def test_residual_chunk_commits_all_of_its_frames(tiny_decoder, tiny_geometry):
    """A short final chunk has no lookahead to hold back."""
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    decoder(_codes(13)[0].transpose(0, 1))
    decoder(_codes(7, seed=11)[0].transpose(0, 1))
    assert decoder.state_frames == 17


def test_stitched_stream_reconstructs_the_whole_sequence_decode(
    tiny_decoder, tiny_geometry
):
    """End-to-end, with lookahead and the worker's splice arithmetic applied:
    the posted chunks concatenate to the whole-sequence decode, sample for
    sample.

    This is the assertion that would catch a 555-sample off-by-one at the
    first seam — the exact class of defect Story 20.4 spent four audition
    rounds finding, and which is invisible to every per-chunk check.
    """
    spf = tiny_geometry.samples_per_frame
    edge = tiny_geometry.edge_loss_samples
    cs, la = 10, 3
    n_frames = 43
    codes = _codes(n_frames)
    with torch.inference_mode():
        whole = tiny_decoder(codes)[0, 0].to(torch.float64).numpy()

    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs + la)
    frames = codes[0].transpose(0, 1)

    posted = []
    tails = []
    index = 0
    while index + cs + la <= n_frames:
        pcm = decoder(frames[index:index + cs + la])
        splice = cs * spf - (edge if index == 0 else 0)
        posted.append(pcm[:splice])
        tails.append(pcm[splice:splice + 16])
        index += cs
    if index < n_frames:
        posted.append(decoder(frames[index:]))

    stitched = np.concatenate(posted).astype(np.float64)
    assert stitched.size == whole.size
    np.testing.assert_allclose(stitched, whole, rtol=0, atol=1e-6)

    # And the retained tails are the next chunk's head: the Story 20.4
    # overlap-add now blends a signal with a copy of itself.
    for i, tail in enumerate(tails[:-1]):
        np.testing.assert_array_equal(tail, posted[i + 1][:tail.size])


def test_overlap_add_is_an_identity_under_carried_state(tiny_decoder, tiny_geometry):
    """AC #3 asks for the Story 20.4 seam blend to be re-evaluated on
    evidence rather than assumed away. This is that evidence.

    The tail a chunk retains past its splice is decoded from the same state
    snapshot the next chunk resumes from, so the two are identical and the
    linear cross-fade between them returns its input unchanged. The blend is
    therefore inert under carried state — neither helping nor harming — which
    is what lets Phase 2 leave it in place and keep the audition to one
    variable.
    """
    from myvoice.services.tts_streaming.streaming_decoder import (
        StreamingDecoderWorker,
    )

    spf = tiny_geometry.samples_per_frame
    edge = tiny_geometry.edge_loss_samples
    cs, la = 10, 3
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs + la)
    frames = _codes(40)[0].transpose(0, 1)

    first = decoder(frames[0:cs + la])
    second = decoder(frames[cs:2 * cs + la])
    splice = cs * spf - edge
    retained = first[splice:splice + 64]

    # The two sides of the blend are bit-identical...
    np.testing.assert_array_equal(retained, second[:retained.size])

    # ...so the cross-fade returns its input, to float32 ULP. The residual
    # below is `a*r + a*(1-r) != a` rounding in the ramp arithmetic itself —
    # 1.5e-08 on 3 of 104 samples, about 130 dB down. It is a property of
    # float32 multiply-add, not a difference between the two renditions:
    # the array_equal assertion above already established there is none.
    blended = StreamingDecoderWorker._apply_overlap_add.__get__(
        _StubWorker(retained)
    )(second)
    np.testing.assert_allclose(blended, second, rtol=0, atol=1e-7)


class _StubWorker:
    """Minimal stand-in exposing only what ``_apply_overlap_add`` reads."""

    def __init__(self, pending):
        self._pending_overlap = np.asarray(pending, dtype=np.float32)


# ============================================================================
# probe_decoder — the build-time gate that must REFUSE rather than degrade
# ============================================================================


def test_probe_accepts_the_real_decoder_graph(tiny_decoder):
    geometry = csc.probe_decoder(tiny_decoder)
    assert geometry.conv_modules > 0
    assert geometry.tconv_modules > 0


def test_probe_refuses_an_unknown_leaf_module(tiny_decoder):
    """A future qwen-tts that inserts a new time-mixing module must fail the
    probe, not be waved through by the traversal's ``return module(x)``
    fallthrough. Passing it through would corrupt audio at every chunk
    boundary — a failure mode that is inaudible as a bug and audible only as
    "the codec got worse"."""
    import copy
    import torch.nn as nn

    forked = copy.deepcopy(tiny_decoder)
    forked.decoder.insert(2, nn.Conv1d(8, 8, kernel_size=3))

    with pytest.raises(csc.UnsupportedDecoderGraph, match="unknown leaf module"):
        csc.probe_decoder(forked)


def test_probe_refuses_a_strided_causal_conv(tiny_decoder, monkeypatch):
    """stride-1 is what makes ``_get_extra_padding_for_conv1d`` return 0 and
    the left-context form exact. A strided causal conv silently breaks that."""
    import copy

    forked = copy.deepcopy(tiny_decoder)
    forked.pre_conv.stride = 2

    with pytest.raises(csc.UnsupportedDecoderGraph, match="stride"):
        csc.probe_decoder(forked)


def test_probe_refuses_a_non_sliding_transformer(tiny_decoder):
    """A full-attention layer would make the carried KV cache grow without
    bound over a long utterance, voiding the AC #2 cost claim."""
    import copy

    forked = copy.deepcopy(tiny_decoder)

    class _Cfg:
        layer_types = ["sliding_attention", "full_attention"]

    forked.pre_transformer.config = _Cfg()
    with pytest.raises(csc.UnsupportedDecoderGraph, match="non-sliding"):
        csc.probe_decoder(forked)


def test_probe_refuses_a_decoder_missing_the_expected_attributes():
    class _Bare:
        pass

    with pytest.raises(csc.UnsupportedDecoderGraph, match="quantizer"):
        csc.probe_decoder(_Bare())


# ============================================================================
# build_stateful_decode_fn — every rejection returns None, never wrong audio
# ============================================================================


def test_build_declines_when_the_kill_switch_is_set(tiny_decoder, monkeypatch):
    """The operator escape hatch, and the mechanism the Phase 3 audition uses
    to render its reference arm from the same build as its candidate arm."""
    monkeypatch.setenv("MYVOICE_CODEC_STATE_CACHE", "0")
    fn, reason = csc.build_stateful_decode_fn(
        tiny_decoder, chunk_size=25, lookahead=5
    )
    assert fn is None
    assert "MYVOICE_CODEC_STATE_CACHE" in reason


@pytest.mark.parametrize("value", ["1", "true", "on", "", "yes"])
def test_kill_switch_only_disables_on_explicit_falsey_values(monkeypatch, value):
    monkeypatch.setenv("MYVOICE_CODEC_STATE_CACHE", value)
    assert csc.state_cache_enabled() is (value.strip().lower() not in
                                         {"0", "false", "no", "off"})


def test_build_declines_when_the_geometry_does_not_match_the_pinned_constants(
    tiny_decoder
):
    """The tiny fixture upsamples 8 samples/frame, not 1920. The builder must
    refuse it rather than ship a decoder whose splice arithmetic disagrees
    with ``streaming_decoder``'s measured constants."""
    fn, reason = csc.build_stateful_decode_fn(
        tiny_decoder, chunk_size=25, lookahead=5
    )
    assert fn is None
    assert "samples/frame" in reason


def test_build_declines_on_an_unsupported_graph_without_raising():
    """Dispatch must never break because the probe was unhappy."""
    class _Bare:
        pass

    fn, reason = csc.build_stateful_decode_fn(
        _Bare(), chunk_size=25, lookahead=5
    )
    assert fn is None
    assert "not supported" in reason


def test_numerical_self_test_passes_on_the_real_decoder(tiny_decoder, tiny_geometry):
    """The build-time gate that runs against the LOADED weights."""
    ok, detail = csc._numerical_self_test(tiny_decoder, tiny_geometry)
    assert ok, detail


def test_numerical_self_test_catches_a_broken_traversal(
    tiny_decoder, tiny_geometry, monkeypatch
):
    """And it must actually be able to fail: with the bias correction removed
    the self-test has to reject the decoder rather than wave it through."""
    def _naive_tconv(module, x, state, key):
        left_pad, right_pad = module.left_pad, module.right_pad
        if left_pad == 0 and right_pad == 0:
            return module.conv(x).contiguous()
        y = module.conv(x)
        prev = state.tconv.get(key)
        if prev is not None:
            y = torch.cat([y[..., :left_pad] + prev, y[..., left_pad:]], dim=-1)
        state.tconv[key] = y[..., y.shape[-1] - right_pad:].clone()
        out = y[..., : y.shape[-1] - right_pad]
        if prev is None:
            out = out[..., left_pad:]
        return out.contiguous()

    monkeypatch.setattr(csc, "_stream_tconv", _naive_tconv)
    ok, detail = csc._numerical_self_test(tiny_decoder, tiny_geometry)
    assert not ok
    assert "NRMSE" in detail


def test_stateful_decoder_advertises_itself_to_the_worker(tiny_decoder, tiny_geometry):
    """``StreamingDecoderWorker`` switches its geometry model on this flag;
    the stock stateless adapter does not carry it."""
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    assert decoder.carries_codec_state is True
    assert callable(decoder.reset)


def test_output_length_mismatch_raises_rather_than_posting_misaligned_audio(
    tiny_decoder, tiny_geometry
):
    """The length identity is a theorem once ``probe_decoder`` has passed, so
    reaching a mismatch means the graph changed mid-session. Say so; do not
    post audio spliced against the wrong arithmetic."""
    wrong = csc.DecoderGeometry(
        samples_per_frame=tiny_geometry.samples_per_frame + 1,
        edge_loss_samples=tiny_geometry.edge_loss_samples,
        conv_modules=1, tconv_modules=1, kv_layers=2,
    )
    decoder = csc.StatefulCodecDecoder(tiny_decoder, wrong, 10, 13)
    with pytest.raises(RuntimeError, match="expected"):
        decoder(_codes(13)[0].transpose(0, 1))


def test_empty_chunk_is_a_no_op(tiny_decoder, tiny_geometry):
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 13)
    out = decoder(torch.zeros((0, 2), dtype=torch.long))
    assert out.size == 0
    assert decoder.state_frames == 0


# ============================================================================
# Story 20.6 — the retired-lookahead geometry
#
# ``codec_state_cache.py`` is NOT modified by Story 20.6. The retirement is
# expressed entirely in the ARGUMENT the dispatch layer builds this decoder
# with: ``lookahead = 0``, so ``window_frames == commit_frames``. The rows
# below pin the two consequences that argument has here — the two-pass decode
# collapses to a single pass, and the Phase 1 regression bars still hold at
# the new geometry — because both are properties this file is the only place
# that can test against a real decoder.
# ============================================================================


def test_build_accepts_the_retired_geometry(tiny_decoder, tiny_geometry):
    """``window_frames == commit_frames`` is legal; only ``<`` is rejected.
    If this ever became an error the retirement would have to be expressed by
    modifying this module instead of by an argument to it."""
    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 10)
    assert decoder.commit_frames == 10
    with pytest.raises(ValueError, match="window_frames"):
        csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 10, 9)


def test_retired_lookahead_decode_is_single_pass(
    tiny_decoder, tiny_geometry, monkeypatch
):
    """AC #1 / Task 2 — the snapshot/restore collapses.

    The two-pass decode exists *solely* to serve the lookahead: commit
    ``chunk_size`` frames on the live state, snapshot, decode the trailing
    lookahead on the snapshot, restore. Story 20.5 measured the tax at
    +7-10 ms/chunk. With ``window_frames == commit_frames`` the commit rule
    takes ``commit = n_frames`` on every chunk, so the second pass is never
    entered — asserted by counting snapshots rather than inferred from timing.
    """
    calls = {"snapshot": 0, "restore": 0}
    real_snapshot = csc.CodecStreamState.snapshot
    real_restore = csc.CodecStreamState.restore

    def _snapshot(self):
        calls["snapshot"] += 1
        return real_snapshot(self)

    def _restore(self, snap):
        calls["restore"] += 1
        return real_restore(self, snap)

    monkeypatch.setattr(csc.CodecStreamState, "snapshot", _snapshot)
    monkeypatch.setattr(csc.CodecStreamState, "restore", _restore)

    cs = 10
    frames = _codes(40)[0].transpose(0, 1)

    # Control: the pre-20.6 geometry pays two passes per full window.
    with_lookahead = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs + 3)
    for k in range(3):
        with_lookahead(frames[k * cs:k * cs + cs + 3])
    assert calls["snapshot"] == 3 and calls["restore"] == 3

    calls["snapshot"] = calls["restore"] = 0
    retired = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs)
    for k in range(3):
        retired(frames[k * cs:(k + 1) * cs])
    assert calls == {"snapshot": 0, "restore": 0}, (
        "the retired geometry still runs the lookahead's two-pass decode; "
        "the producer-throughput half of Story 20.6 is not being realised"
    )


def test_retired_lookahead_stitched_stream_reconstructs_the_whole_decode(
    tiny_decoder, tiny_geometry
):
    """The Phase 1 regression bar, at the new geometry.

    With no lookahead there is no splice arithmetic left: the worker posts
    every decode whole. The posted chunks must still concatenate into the
    whole-sequence decode sample for sample, first chunk short by the edge
    loss and every later one frame-exact.
    """
    spf = tiny_geometry.samples_per_frame
    edge = tiny_geometry.edge_loss_samples
    cs = 10
    n_frames = 43
    codes = _codes(n_frames)
    with torch.inference_mode():
        whole = tiny_decoder(codes)[0, 0].to(torch.float64).numpy()

    decoder = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs)
    frames = codes[0].transpose(0, 1)

    posted = []
    index = 0
    while index < n_frames:
        take = min(cs, n_frames - index)
        pcm = decoder(frames[index:index + take])
        expected = spf * take - (edge if index == 0 else 0)
        assert pcm.size == expected, (
            f"chunk at frame {index} returned {pcm.size} samples, expected "
            f"{expected} — the edge loss moved"
        )
        posted.append(pcm)
        index += take

    stitched = np.concatenate(posted).astype(np.float64)
    assert stitched.size == whole.size == spf * n_frames - edge
    np.testing.assert_allclose(stitched, whole, rtol=0, atol=1e-6)


def test_retired_lookahead_matches_the_lookahead_geometrys_output(
    tiny_decoder, tiny_geometry
):
    """The audible claim, made exactly.

    Retiring the lookahead is meant to change *what work is done*, not *what
    audio is posted*: the trim removed precisely the lookahead's worth of PCM,
    and under carried state the retained tail was bit-identical to the next
    chunk's head, so the blend was an identity. The two geometries must
    therefore produce the same posted stream — which is what makes the AC #4
    audition a test of that claim rather than of an unknown.
    """
    spf = tiny_geometry.samples_per_frame
    edge = tiny_geometry.edge_loss_samples
    cs, la = 10, 3
    n_frames = 43
    codes = _codes(n_frames)
    frames = codes[0].transpose(0, 1)

    with_la = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs + la)
    posted_la = []
    index = 0
    while index + cs + la <= n_frames:
        pcm = with_la(frames[index:index + cs + la])
        splice = cs * spf - (edge if index == 0 else 0)
        posted_la.append(pcm[:splice])
        index += cs
    if index < n_frames:
        posted_la.append(with_la(frames[index:]))

    retired = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs)
    posted_retired = []
    index = 0
    while index < n_frames:
        posted_retired.append(retired(frames[index:index + cs]))
        index += cs

    a = np.concatenate(posted_la).astype(np.float64)
    b = np.concatenate(posted_retired).astype(np.float64)
    assert a.size == b.size
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_worker_refuses_a_decode_fn_built_for_a_different_window(
    tiny_decoder, tiny_geometry
):
    """Story 20.6 — the one silent-wrong-audio path the conditional
    retirement introduces.

    A ``StatefulCodecDecoder`` built for a 25-frame window, handed 30-frame
    chunks, fails its own ``window_frames > commit_frames`` guard and commits
    all 30 — so the next chunk resumes five frames in its own future and every
    seam skips 400 ms of speech. Nothing raises, and the length identity is
    still satisfied, so the worker's geometry trip-wire stays quiet: exactly
    the defect class Story 20.4 spent four audition rounds locating.

    The dispatch layer cannot produce the mismatch (it derives the streamer
    geometry and the decode_fn from one ``carries_codec_state`` read, and a
    source invariant pins that). This row is the belt for every other caller.
    """
    from myvoice.services.tts_streaming import CodecTokenStreamer
    from myvoice.services.tts_streaming.streaming_decoder import (
        StreamingDecoderWorker,
    )

    retired = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 25, 25)

    # The mismatch: a retired decode_fn behind an un-retired streamer.
    with pytest.raises(ValueError, match="window"):
        StreamingDecoderWorker(
            streamer=CodecTokenStreamer(chunk_size=25, lookahead=5),
            decode_fn=retired, post_mutation=lambda *a: None,
            session_id="s",
        )

    # And the matching pair is accepted, both ways round.
    matched = CodecTokenStreamer(chunk_size=25, lookahead=5)
    matched.apply_codec_state_geometry(True)
    StreamingDecoderWorker(
        streamer=matched, decode_fn=retired,
        post_mutation=lambda *a: None, session_id="s",
    )
    with_lookahead = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, 25, 30)
    StreamingDecoderWorker(
        streamer=CodecTokenStreamer(chunk_size=25, lookahead=5),
        decode_fn=with_lookahead, post_mutation=lambda *a: None,
        session_id="s",
    )


def test_the_mismatch_the_worker_refuses_really_does_desync(
    tiny_decoder, tiny_geometry
):
    """The row above is only worth having if the thing it refuses is
    genuinely broken. Demonstrate the desync directly, so the guard is
    justified in executable form rather than by assertion."""
    cs, la = 10, 3
    frames = _codes(40)[0].transpose(0, 1)

    retired = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs)
    retired(frames[0:cs + la])          # a full pre-20.6 window
    assert retired.state_frames == cs + la, (
        "a decode_fn built for the retired window did not over-commit on a "
        "wider chunk; this test's premise no longer holds and the worker "
        "guard above may be unnecessary."
    )

    matched = csc.StatefulCodecDecoder(tiny_decoder, tiny_geometry, cs, cs + la)
    matched(frames[0:cs + la])
    assert matched.state_frames == cs, (
        "the correctly-paired decoder must commit only to the splice"
    )
