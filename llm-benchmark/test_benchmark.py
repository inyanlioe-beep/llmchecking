"""Tests for benchmark core logic (no network calls)."""
from unittest.mock import patch, MagicMock
from benchmark import LLMConfig, run_parallel_benchmarks, judge_quality, BenchmarkResult


def _mock_result(name: str, success=True, error=None) -> BenchmarkResult:
    r = BenchmarkResult(llm_name=name, model="mock", prompt="test")
    if success:
        r.response_text = "Paris."
        r.ttft_ms = 50.0
        r.total_latency_s = 0.5
        r.output_tokens = 10
        r.tokens_per_second = 20.0
    else:
        r.success = False
        r.error = error or "timeout"
    return r


def test_parallel_returns_all_results():
    cfg1 = LLMConfig("A", "http://a", "key", "model-a")
    cfg2 = LLMConfig("B", "http://b", "key", "model-b")
    with patch("benchmark.run_single_benchmark", side_effect=[
        _mock_result("A"), _mock_result("B")
    ]):
        results = run_parallel_benchmarks([cfg1, cfg2], "prompt")
    assert len(results) == 2
    assert {r.llm_name for r in results} == {"A", "B"}


def test_parallel_handles_one_failure():
    cfg1 = LLMConfig("A", "http://a", "key", "model-a")
    cfg2 = LLMConfig("B", "http://b", "key", "model-b")
    with patch("benchmark.run_single_benchmark", side_effect=[
        _mock_result("A"), _mock_result("B", success=False, error="timeout")
    ]):
        results = run_parallel_benchmarks([cfg1, cfg2], "prompt")
    assert len(results) == 2
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    assert len(successes) == 1
    assert len(failures) == 1


def test_judge_quality_parses_score():
    judge_cfg = LLMConfig("Judge", "http://j", "key", "judge-model")
    scored = _mock_result("Judge")
    scored.response_text = "8"
    with patch("benchmark.run_single_benchmark", return_value=scored):
        score = judge_quality(judge_cfg, "prompt", "response")
    assert score == 8.0


def test_judge_quality_returns_none_on_error():
    judge_cfg = LLMConfig("Judge", "http://j", "key", "judge-model")
    fail = _mock_result("Judge", success=False)
    with patch("benchmark.run_single_benchmark", return_value=fail):
        score = judge_quality(judge_cfg, "prompt", "response")
    assert score is None


def test_conversation_round_robin_order():
    cfg1 = LLMConfig("Alpha", "http://a", "key", "model-a")
    cfg2 = LLMConfig("Beta", "http://b", "key", "model-b")

    def fake_call(config, messages, timeout=60):
        return f"Reply from {config.name}", None

    with patch("benchmark._call_llm_with_history", side_effect=fake_call):
        from benchmark import run_llm_conversation
        log = run_llm_conversation([cfg1, cfg2], "Hello", num_turns=4)

    # turn 0 = Human opening, turns 1-4 alternate Alpha/Beta
    speakers = [m.llm_name for m in log]
    assert speakers[0] == "Human"
    assert speakers[1] == "Alpha"
    assert speakers[2] == "Beta"
    assert speakers[3] == "Alpha"
    assert speakers[4] == "Beta"


def test_conversation_stops_on_error():
    cfg1 = LLMConfig("Alpha", "http://a", "key", "model-a")
    cfg2 = LLMConfig("Beta", "http://b", "key", "model-b")

    call_count = 0
    def fake_call(config, messages, timeout=60):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "", "timeout error"  # first LLM fails
        return "ok", None

    with patch("benchmark._call_llm_with_history", side_effect=fake_call):
        from benchmark import run_llm_conversation
        log = run_llm_conversation([cfg1, cfg2], "Hello", num_turns=6)

    # should stop after first error (Human + 1 error message)
    assert len(log) == 2
    assert log[1].error == "timeout error"
    assert log[1].content == ""
