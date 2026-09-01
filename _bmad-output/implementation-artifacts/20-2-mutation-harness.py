"""Mutation harness for Story 20.2 review fixes.

For each fix, revert it in the source, run the story test files, and report
whether the suite goes red. A fix whose reversion keeps the suite green is a
fix with no test behind it.
"""
import subprocess
import sys

P = 'src/myvoice/services/qwen_tts_service.py'
import os
PY = os.path.abspath('python310/python.exe')
TESTS = [
    'tests/unit/services/test_compile_priming_audio_suppression.py',
    'tests/unit/services/test_qwen_tts_service_compile_warmup.py',
]

MUTATIONS = {
    'F1 play_dual_stream ungated': (
        "                    and sid is not None\n"
        "                    and not suppressed\n"
        "                    and not self._cancel_requested\n",
        "                    and sid is not None\n"
        "                    and not self._cancel_requested\n",
    ),
    'F3 cache write ungated (TRUE_STREAM)': (
        "            audio_file = (\n"
        "                None if suppressed\n"
        "                else self._save_audio_to_cache(complete_audio, sample_rate)\n"
        "            )\n"
        "\n"
        "            generation_time = time.time() - start_time\n"
        "            self._successful_requests += 1\n"
        "            self._last_generation_time = generation_time\n"
        "            self._generation_state = GenerationState.COMPLETE\n"
        "\n"
        "            if first_chunk_time is not None:",
        "            audio_file = self._save_audio_to_cache(complete_audio, sample_rate)\n"
        "\n"
        "            generation_time = time.time() - start_time\n"
        "            self._successful_requests += 1\n"
        "            self._last_generation_time = generation_time\n"
        "            self._generation_state = GenerationState.COMPLETE\n"
        "\n"
        "            if first_chunk_time is not None:",
    ),
    'F4 registry session ungated (TRUE_STREAM)': (
        "        if self._session_registry is not None and not suppressed:\n"
        "            sid = self._session_registry.create_session(\n"
        "                text=request.text,\n"
        "                voice=self._resolve_voice_label(request),\n"
        "                model_type=self._resolve_model_type_label(request),\n"
        "                source=SessionSource.GENERATED,\n"
        "                session_id=request.session_id,\n"
        "            )\n"
        "            # Story 16.5: publish the active session id so cancel_generation\n"
        "            # can request_cancel(sid) -> hook -> streamer._cancel_event.set().\n"
        "            self._current_session_id = sid",
        "        if self._session_registry is not None:\n"
        "            sid = self._session_registry.create_session(\n"
        "                text=request.text,\n"
        "                voice=self._resolve_voice_label(request),\n"
        "                model_type=self._resolve_model_type_label(request),\n"
        "                source=SessionSource.GENERATED,\n"
        "                session_id=request.session_id,\n"
        "            )\n"
        "            # Story 16.5: publish the active session id so cancel_generation\n"
        "            # can request_cancel(sid) -> hook -> streamer._cancel_event.set().\n"
        "            self._current_session_id = sid",
    ),
    'F2 finally clears unconditionally (TRUE_STREAM)': (
        "            if (\n"
        "                owned_generation_task is not None\n"
        "                and self._current_generation_task is owned_generation_task\n"
        "            ):\n"
        "                self._current_generation_task = None\n"
        "            if sid is not None and self._current_session_id == sid:\n"
        "                self._current_session_id = None\n"
        "\n"
        "    @staticmethod\n"
        "    def _fallback_chain_from",
        "            self._current_generation_task = None\n"
        "            self._current_session_id = None\n"
        "\n"
        "    @staticmethod\n"
        "    def _fallback_chain_from",
    ),
    'F2b prime resets _cancel_requested': (
        "        suppressed = self._is_suppressed(request)\n"
        "        if not suppressed:\n"
        "            # A suppressed (compile-priming) generation must NOT reset the\n",
        "        suppressed = self._is_suppressed(request)\n"
        "        self._cancel_requested = False\n"
        "        if False:\n"
        "            # A suppressed (compile-priming) generation must NOT reset the\n",
    ),
    'sink returns the callback regardless': (
        "        if self._is_suppressed(request):\n"
        "            return None\n"
        "        return self._audio_chunk_ready_callback",
        "        return self._audio_chunk_ready_callback",
    ),
    'F7 indicator cleared unconditionally': (
        "        if self._last_preparing_voice_message == message:\n"
        "            self._emit_preparing_voice(None)",
        "        self._emit_preparing_voice(None)",
    ),
    'F6 trip-wire removed': (
        "        if not self._is_suppressed(request):\n"
        "            raise RuntimeError(\n"
        "                \"compile-priming request is not suppressed - refusing to \"\n"
        "                \"dispatch a generation that could reach the user's \"\n"
        "                \"speakers (Story 20.2 AC #2)\"\n"
        "            )\n"
        "        assert self._audio_chunk_sink(request) is None\n",
        "",
    ),
}

original = open(P, encoding='utf-8').read()
rows = []
try:
    for name, (old, new) in MUTATIONS.items():
        if original.count(old) != 1:
            rows.append((name, 'SKIP', 'anchor count=%d' % original.count(old)))
            continue
        open(P, 'w', encoding='utf-8').write(original.replace(old, new))
        r = subprocess.run(
            [PY, '-m', 'pytest', *TESTS, '-q', '--no-header'],
            capture_output=True, text=True)
        tail = [ln for ln in r.stdout.splitlines() if 'passed' in ln or 'failed' in ln or 'error' in ln]
        verdict = 'CAUGHT' if r.returncode != 0 else 'MISSED'
        rows.append((name, verdict, tail[-1].strip() if tail else '?'))
finally:
    open(P, 'w', encoding='utf-8').write(original)

print()
for name, verdict, detail in rows:
    print('%-46s %-7s %s' % (name, verdict, detail))
missed = [r for r in rows if r[1] != 'CAUGHT']
print()
print('MISSED/SKIPPED:', len(missed))
sys.exit(1 if missed else 0)
