"""Optional LLM backends + a token/cost meter.

The lab runs fully offline by default (`backend="none"`), in which case the
"answer" is simply the textualised retrieval context plus a short extractive
summary. If an OpenAI or Anthropic key is configured, the same context is sent
to a real model for natural-language synthesis.

A :class:`TokenMeter` records token usage and wall-clock time so Deliverable #4
(cost analysis) can be produced for both indexing and querying.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when no tokenizer is available."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


@dataclass
class TokenMeter:
    """Accumulates token usage + timing, grouped by stage label."""

    prompt_tokens: Dict[str, int] = field(default_factory=dict)
    completion_tokens: Dict[str, int] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)

    def add(self, stage: str, prompt: int = 0, completion: int = 0, seconds: float = 0.0):
        self.prompt_tokens[stage] = self.prompt_tokens.get(stage, 0) + int(prompt)
        self.completion_tokens[stage] = self.completion_tokens.get(stage, 0) + int(completion)
        self.seconds[stage] = self.seconds.get(stage, 0.0) + float(seconds)
        self.calls[stage] = self.calls.get(stage, 0) + 1

    def total_tokens(self) -> int:
        return sum(self.prompt_tokens.values()) + sum(self.completion_tokens.values())

    def as_rows(self) -> List[dict]:
        rows = []
        stages = set(self.prompt_tokens) | set(self.completion_tokens) | set(self.seconds)
        for s in sorted(stages):
            pt = self.prompt_tokens.get(s, 0)
            ct = self.completion_tokens.get(s, 0)
            rows.append(
                {
                    "stage": s,
                    "calls": self.calls.get(s, 0),
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                    "seconds": round(self.seconds.get(s, 0.0), 3),
                }
            )
        return rows

    def report(self) -> str:
        lines = ["=== Cost / Token usage report ==="]
        for r in self.as_rows():
            lines.append(
                f"  {r['stage']:<22} calls={r['calls']:<4} "
                f"tok={r['total_tokens']:<8} (p={r['prompt_tokens']}, c={r['completion_tokens']}) "
                f"time={r['seconds']}s"
            )
        lines.append(f"  {'TOTAL':<22} tok={self.total_tokens()}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM wrapper
# --------------------------------------------------------------------------- #
class LLM:
    """Unified small wrapper over OpenAI / Anthropic / offline."""

    def __init__(self, backend: str = "none", model: str = "", meter: Optional[TokenMeter] = None):
        self.backend = backend
        self.model = model
        self.meter = meter or TokenMeter()
        self._client = None
        if backend == "openai":
            from openai import OpenAI  # noqa: lazy import

            self._client = OpenAI()
            self.model = model or "gpt-4o-mini"
        elif backend == "anthropic":
            import anthropic  # noqa: lazy import

            self._client = anthropic.Anthropic()
            self.model = model or "claude-haiku-4-5-20251001"

    # ---- generic completion ------------------------------------------------
    def complete(self, system: str, user: str, stage: str = "llm", max_tokens: int = 600) -> str:
        t0 = time.perf_counter()
        if self.backend == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            self.meter.add(
                stage,
                prompt=getattr(usage, "prompt_tokens", estimate_tokens(system + user)),
                completion=getattr(usage, "completion_tokens", estimate_tokens(out)),
                seconds=time.perf_counter() - t0,
            )
            return out.strip()

        if self.backend == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            out = "".join(getattr(b, "text", "") for b in resp.content)
            usage = getattr(resp, "usage", None)
            self.meter.add(
                stage,
                prompt=getattr(usage, "input_tokens", estimate_tokens(system + user)),
                completion=getattr(usage, "output_tokens", estimate_tokens(out)),
                seconds=time.perf_counter() - t0,
            )
            return out.strip()

        # ---- offline fallback: extractive answer over the context ---------
        out = _offline_answer(user)
        self.meter.add(
            stage,
            prompt=estimate_tokens(system + user),
            completion=estimate_tokens(out),
            seconds=time.perf_counter() - t0,
        )
        return out


def _offline_answer(user_prompt: str) -> str:
    """Deterministic extractive 'answer' used when no API key is present.

    We pull the question + context out of the prompt and return the 3 context
    sentences most lexically similar to the question. This is intentionally
    simple — its purpose is to let the *pipeline* run end-to-end offline, not to
    rival a real LLM.
    """
    q = ""
    ctx = user_prompt
    if "Question:" in user_prompt:
        after = user_prompt.split("Question:", 1)[1]
        q = after.split("\n", 1)[0].strip()
    if "Context:" in user_prompt:
        ctx = user_prompt.split("Context:", 1)[1]

    q_terms = set(re.findall(r"[a-z0-9]+", q.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", ctx)
    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        terms = set(re.findall(r"[a-z0-9]+", s.lower()))
        overlap = len(q_terms & terms)
        if overlap:
            scored.append((overlap, s))
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:3]]
    if not top:
        return "[offline] No directly relevant sentence found in the retrieved context."
    return "[offline extractive answer] " + " ".join(top)


def get_llm(cfg, meter: Optional[TokenMeter] = None) -> LLM:
    """Build an LLM, auto-downgrading to offline if a key is missing."""
    backend = cfg.llm_backend
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        backend = "none"
    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        backend = "none"
    model = cfg.openai_model if backend == "openai" else cfg.anthropic_model
    return LLM(backend=backend, model=model, meter=meter)
