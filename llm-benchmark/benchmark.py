"""
Core benchmark runner: fires requests to multiple LLMs in parallel,
measures TTFT, total latency, tokens/s, and optionally quality score.
Also provides LLM-to-LLM conversation (round-robin).
"""
from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from openai import OpenAI


@dataclass
class LLMConfig:
    name: str           # display name
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class BenchmarkResult:
    llm_name: str
    model: str
    prompt: str
    response_text: str = ""
    ttft_ms: float = 0.0          # time to first token
    total_latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    quality_score: Optional[float] = None
    error: Optional[str] = None
    success: bool = True


def run_single_benchmark(config: LLMConfig, prompt: str, timeout: int = 60) -> BenchmarkResult:
    result = BenchmarkResult(llm_name=config.name, model=config.model, prompt=prompt)
    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=timeout,
    )

    first_token_time: Optional[float] = None
    chunks: list[str] = []
    start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
                result.ttft_ms = (first_token_time - start) * 1000

            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)

            # grab usage from last chunk if available
            if hasattr(chunk, "usage") and chunk.usage:
                result.input_tokens = chunk.usage.prompt_tokens or 0
                result.output_tokens = chunk.usage.completion_tokens or 0

        end = time.perf_counter()
        result.total_latency_s = end - start
        result.response_text = "".join(chunks)

        # estimate output tokens from words if API didn't return usage
        if result.output_tokens == 0:
            result.output_tokens = max(1, len(result.response_text.split()) * 4 // 3)

        elapsed = result.total_latency_s - (result.ttft_ms / 1000)
        result.tokens_per_second = result.output_tokens / max(elapsed, 0.001)

    except Exception as exc:  # noqa: BLE001
        result.success = False
        result.error = str(exc)
        result.total_latency_s = time.perf_counter() - start

    return result


def run_parallel_benchmarks(
    configs: list[LLMConfig],
    prompt: str,
    timeout: int = 60,
) -> list[BenchmarkResult]:
    """Run all LLMs concurrently using threads (one per LLM)."""
    results: list[Optional[BenchmarkResult]] = [None] * len(configs)

    def worker(idx: int, cfg: LLMConfig) -> None:
        results[idx] = run_single_benchmark(cfg, prompt, timeout)

    threads = [
        threading.Thread(target=worker, args=(i, cfg), daemon=True)
        for i, cfg in enumerate(configs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout + 5)

    return [r for r in results if r is not None]


def judge_quality(
    judge_config: LLMConfig,
    prompt: str,
    response: str,
    timeout: int = 30,
) -> Optional[float]:
    """Ask a judge LLM to score response quality 1-10."""
    judge_prompt = (
        f"Rate the following AI response to the given prompt on a scale of 1 to 10 "
        f"based on relevance, accuracy, and clarity. "
        f"Return ONLY a single integer number (1-10), nothing else.\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response: {response}"
    )
    result = run_single_benchmark(judge_config, judge_prompt, timeout)
    if not result.success:
        return None
    try:
        m = re.search(r"\b([1-9]|10)\b", result.response_text)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


# ─── LLM-to-LLM conversation ─────────────────────────────────────────────────

@dataclass
class ChatMessage:
    llm_name: str
    model: str
    role: str          # "user" (opening) or "assistant"
    content: str
    turn: int          # 0-based turn index
    error: Optional[str] = None


def _call_llm_with_history(
    config: LLMConfig,
    messages: list[dict],
    timeout: int = 60,
) -> tuple[str, Optional[str]]:
    """Send a full message history to an LLM and return (response_text, error)."""
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=timeout)
    chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
        return "".join(chunks), None
    except Exception as exc:  # noqa: BLE001
        return "", str(exc)


def run_llm_conversation(
    configs: list[LLMConfig],
    opening_message: str,
    num_turns: int = 6,
    system_prompt: str = "",
    timeout: int = 60,
    on_message: Optional[Callable[[ChatMessage], None]] = None,
) -> list[ChatMessage]:
    """
    Round-robin conversation between LLMs.

    Each LLM receives the full shared history so far (from its own perspective
    as "assistant" and everyone else as "user"), then replies.
    The reply is appended to the shared history and the next LLM takes its turn.

    on_message: called after each turn so callers can stream updates to the UI.
    """
    if len(configs) < 2:
        raise ValueError("Need at least 2 LLMs for a conversation.")

    log: list[ChatMessage] = []

    # Shared chronological history as plain dicts: {speaker, content}
    shared: list[dict] = [{"speaker": "Human", "content": opening_message}]

    # Emit the opening message
    opening = ChatMessage(
        llm_name="Human", model="", role="user",
        content=opening_message, turn=0,
    )
    log.append(opening)
    if on_message:
        on_message(opening)

    for turn in range(num_turns):
        speaker_cfg = configs[turn % len(configs)]

        # Build messages[] from shared history from this LLM's POV:
        # messages it sent → "assistant", everything else → "user"
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for entry in shared:
            role = "assistant" if entry["speaker"] == speaker_cfg.name else "user"
            messages.append({"role": role, "content": entry["content"]})

        response, error = _call_llm_with_history(speaker_cfg, messages, timeout)

        msg = ChatMessage(
            llm_name=speaker_cfg.name,
            model=speaker_cfg.model,
            role="assistant",
            content=response if not error else "",
            turn=turn + 1,
            error=error,
        )
        log.append(msg)
        if on_message:
            on_message(msg)

        if error:
            break  # stop conversation on hard error

        shared.append({"speaker": speaker_cfg.name, "content": response})

    return log
