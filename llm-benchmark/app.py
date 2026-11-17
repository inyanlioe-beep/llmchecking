"""
Streamlit app: LLM Benchmark — parallel testing of multiple OpenAI-compatible LLMs.
"""
from __future__ import annotations

import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from benchmark import LLMConfig, BenchmarkResult, run_parallel_benchmarks, judge_quality, run_llm_conversation, ChatMessage

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── session state init ───────────────────────────────────────────────────────
if "llm_configs" not in st.session_state:
    st.session_state.llm_configs: list[dict] = []
if "history" not in st.session_state:
    st.session_state.history: list[list[dict]] = []
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs: list[list[dict]] = []  # list of past conversation logs


# ─── helpers ─────────────────────────────────────────────────────────────────
def configs_to_obj(raw: list[dict]) -> list[LLMConfig]:
    return [
        LLMConfig(
            name=c["name"],
            base_url=c["base_url"].rstrip("/"),
            api_key=c["api_key"],
            model=c["model"],
            temperature=c.get("temperature", 0.7),
            max_tokens=c.get("max_tokens", 512),
        )
        for c in raw
        if c.get("base_url") and c.get("model")
    ]


def result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "LLM": r.llm_name,
        "Model": r.model,
        "Success": r.success,
        "TTFT (ms)": round(r.ttft_ms, 1) if r.success else None,
        "Latency (s)": round(r.total_latency_s, 3) if r.success else None,
        "Tokens/s": round(r.tokens_per_second, 1) if r.success else None,
        "Output tokens": r.output_tokens if r.success else None,
        "Quality (1-10)": r.quality_score,
        "Response": r.response_text if r.success else r.error,
    }


# ─── sidebar: LLM config ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ LLM Benchmark")
    st.caption("Parallel testing of OpenAI-compatible APIs")

    st.header("🔧 LLM Providers")

    # render existing configs
    to_delete = []
    for i, cfg in enumerate(st.session_state.llm_configs):
        with st.expander(f"**{cfg.get('name', f'LLM {i+1}')}**", expanded=False):
            cfg["name"] = st.text_input("Display name", cfg.get("name", f"LLM {i+1}"), key=f"name_{i}")
            cfg["base_url"] = st.text_input("Base URL", cfg.get("base_url", ""), key=f"url_{i}",
                                             placeholder="https://api.openai.com/v1")
            cfg["api_key"] = st.text_input("API Key", cfg.get("api_key", ""), key=f"key_{i}",
                                            type="password", placeholder="sk-...")
            cfg["model"] = st.text_input("Model name", cfg.get("model", ""), key=f"model_{i}",
                                          placeholder="gpt-4o-mini")
            col1, col2 = st.columns(2)
            cfg["temperature"] = col1.slider("Temp", 0.0, 2.0, cfg.get("temperature", 0.7),
                                              0.05, key=f"temp_{i}")
            cfg["max_tokens"] = col2.number_input("Max tokens", 64, 4096,
                                                   cfg.get("max_tokens", 512), 64, key=f"tok_{i}")
            if st.button("🗑 Remove", key=f"del_{i}"):
                to_delete.append(i)

    for idx in reversed(to_delete):
        st.session_state.llm_configs.pop(idx)
    if to_delete:
        st.rerun()

    if st.button("➕ Add LLM", use_container_width=True):
        st.session_state.llm_configs.append({
            "name": f"LLM {len(st.session_state.llm_configs)+1}",
            "base_url": "",
            "api_key": "",
            "model": "",
            "temperature": 0.7,
            "max_tokens": 512,
        })
        st.rerun()

    st.divider()
    st.header("⚖️ Judge LLM (optional)")
    use_judge = st.toggle("Enable quality scoring", value=False)
    judge_cfg_raw: dict = {}
    if use_judge:
        judge_cfg_raw["base_url"] = st.text_input("Judge base URL", key="judge_url",
                                                    placeholder="https://api.openai.com/v1")
        judge_cfg_raw["api_key"] = st.text_input("Judge API Key", key="judge_key",
                                                   type="password", placeholder="sk-...")
        judge_cfg_raw["model"] = st.text_input("Judge model", key="judge_model",
                                                placeholder="gpt-4o-mini")

    st.divider()
    timeout = st.number_input("Timeout per LLM (s)", 10, 300, 60, 10)


# ─── main area: top-level mode tabs ──────────────────────────────────────────
main_tab_benchmark, main_tab_chat = st.tabs(["⚡ Benchmark", "🗣️ LLM Chat"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
with main_tab_benchmark:
    st.header("📝 Benchmark Prompt")

    prompt_mode = st.radio("Prompt mode", ["Single prompt", "Prompt set"], horizontal=True)

    prompts: list[str] = []
    if prompt_mode == "Single prompt":
        p = st.text_area("Enter prompt", height=120,
                          placeholder="Explain the difference between TCP and UDP in simple terms.")
        if p.strip():
            prompts = [p.strip()]
    else:
        raw_prompts = st.text_area(
            "Enter multiple prompts (one per line)",
            height=200,
            placeholder="What is photosynthesis?\nExplain quantum entanglement.\nWrite a haiku about code.",
        )
        prompts = [l.strip() for l in raw_prompts.splitlines() if l.strip()]
        if prompts:
            st.caption(f"{len(prompts)} prompt(s) loaded.")

    col_run, col_clear = st.columns([3, 1])
    run_clicked = col_run.button(
        "▶ Run Benchmark",
        use_container_width=True,
        type="primary",
        disabled=len(st.session_state.llm_configs) < 2 or not prompts,
    )
    if col_clear.button("🗑 Clear history", use_container_width=True):
        st.session_state.history.clear()
        st.rerun()

    if len(st.session_state.llm_configs) < 2:
        st.info("Add at least 2 LLM providers in the sidebar to run a benchmark.")

    if run_clicked and prompts:
        configs = configs_to_obj(st.session_state.llm_configs)
        if len(configs) < 2:
            st.error("Please fill in Base URL and Model for at least 2 LLMs.")
        else:
            judge_config: LLMConfig | None = None
            if use_judge and judge_cfg_raw.get("base_url") and judge_cfg_raw.get("model"):
                judge_config = LLMConfig(
                    name="Judge",
                    base_url=judge_cfg_raw["base_url"].rstrip("/"),
                    api_key=judge_cfg_raw.get("api_key", ""),
                    model=judge_cfg_raw["model"],
                    max_tokens=16,
                )

            all_run_results: list[dict] = []

            progress = st.progress(0, text="Running benchmarks…")
            for pi, prompt in enumerate(prompts):
                with st.spinner(f"Prompt {pi+1}/{len(prompts)}: running {len(configs)} LLMs in parallel…"):
                    results = run_parallel_benchmarks(configs, prompt, timeout=int(timeout))

                if judge_config:
                    with st.spinner("Scoring quality with judge LLM…"):
                        for r in results:
                            if r.success:
                                r.quality_score = judge_quality(judge_config, prompt, r.response_text)

                all_run_results.extend([result_to_dict(r) for r in results])
                progress.progress((pi + 1) / len(prompts),
                                  text=f"Done {pi+1}/{len(prompts)} prompts")

            st.session_state.history.append(all_run_results)
            progress.empty()
            st.success("Benchmark complete!")

    if st.session_state.history:
        latest = st.session_state.history[-1]
        df = pd.DataFrame(latest)

        st.header("📊 Latest Results")
        tabs = st.tabs(["📈 Charts", "📋 Table", "💬 Responses"])

        with tabs[0]:
            successful = df[df["Success"] == True].copy()  # noqa: E712
            if successful.empty:
                st.warning("No successful responses to chart.")
            else:
                c1, c2 = st.columns(2)

                with c1:
                    fig_latency = px.bar(
                        successful, x="LLM", y="Latency (s)",
                        color="LLM", title="Total Latency (s)", text_auto=".2f",
                    )
                    fig_latency.update_traces(textposition="outside")
                    st.plotly_chart(fig_latency, use_container_width=True)

                with c2:
                    fig_tps = px.bar(
                        successful, x="LLM", y="Tokens/s",
                        color="LLM", title="Throughput (Tokens/s)", text_auto=".1f",
                    )
                    fig_tps.update_traces(textposition="outside")
                    st.plotly_chart(fig_tps, use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    fig_ttft = px.bar(
                        successful, x="LLM", y="TTFT (ms)",
                        color="LLM", title="Time to First Token (ms)", text_auto=".0f",
                    )
                    fig_ttft.update_traces(textposition="outside")
                    st.plotly_chart(fig_ttft, use_container_width=True)

                with c4:
                    if successful["Quality (1-10)"].notna().any():
                        fig_q = px.bar(
                            successful.dropna(subset=["Quality (1-10)"]),
                            x="LLM", y="Quality (1-10)",
                            color="LLM", title="Quality Score (1-10)",
                            text_auto=".1f", range_y=[0, 10],
                        )
                        fig_q.update_traces(textposition="outside")
                        st.plotly_chart(fig_q, use_container_width=True)
                    else:
                        st.info("Enable Judge LLM to see quality scores.")

        with tabs[1]:
            display_cols = [c for c in df.columns if c != "Response"]
            st.dataframe(df[display_cols], use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Export CSV", csv,
                file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with tabs[2]:
            for row in latest:
                with st.expander(f"**{row['LLM']}** — {row['Model']}", expanded=False):
                    if row["Success"]:
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Latency", f"{row['Latency (s)']}s")
                        mc2.metric("Tokens/s", row["Tokens/s"])
                        mc3.metric("TTFT", f"{row['TTFT (ms)']}ms")
                        st.markdown("**Response:**")
                        st.write(row["Response"])
                    else:
                        st.error(f"Error: {row['Response']}")

        if len(st.session_state.history) > 1:
            st.header(f"🕓 History ({len(st.session_state.history)} runs)")
            for i, run in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Run {len(st.session_state.history)-i}", expanded=False):
                    st.dataframe(
                        pd.DataFrame(run)[[c for c in pd.DataFrame(run).columns if c != "Response"]],
                        use_container_width=True,
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — LLM CHAT
# ════════════════════════════════════════════════════════════════════════════

# colour palette for chat bubbles — cycles through LLMs
_BUBBLE_COLORS = ["#4F8EF7", "#E8503A", "#2DBF76", "#F5A623", "#9B59B6", "#1ABC9C"]

def _llm_color(name: str, configs: list[dict]) -> str:
    names = [c.get("name", "") for c in configs]
    idx = names.index(name) if name in names else 0
    return _BUBBLE_COLORS[idx % len(_BUBBLE_COLORS)]


def _render_chat_log(log: list[dict], llm_cfgs: list[dict]) -> None:
    """Render stored chat log (list of dicts) as chat bubbles."""
    for msg in log:
        is_human = msg["llm_name"] == "Human"
        with st.chat_message("user" if is_human else "assistant",
                              avatar="🧑" if is_human else "🤖"):
            label = f"**{msg['llm_name']}**" + (f" `{msg['model']}`" if msg.get("model") else "")
            st.markdown(label)
            if msg.get("error"):
                st.error(msg["error"])
            else:
                st.write(msg["content"])


with main_tab_chat:
    st.header("🗣️ LLM Chat — Let LLMs talk to each other")
    st.caption(
        "LLMs take turns responding in round-robin order. "
        "Each LLM sees the full conversation history from its own perspective."
    )

    chat_configs = configs_to_obj(st.session_state.llm_configs)

    if len(chat_configs) < 2:
        st.info("Add at least 2 LLM providers in the sidebar to start a conversation.")
    else:
        # ── settings row ──
        cc1, cc2, cc3 = st.columns([3, 1, 1])
        chat_topic = cc1.text_input(
            "Opening message / topic",
            placeholder="Debate: Is Python better than JavaScript for backend development?",
        )
        num_turns = cc2.number_input("Turns", min_value=2, max_value=30, value=6, step=2)
        chat_timeout = cc3.number_input("Timeout (s)", min_value=10, max_value=300, value=60, step=10)

        with st.expander("⚙️ System prompt (optional — applies to all LLMs)", expanded=False):
            system_prompt = st.text_area(
                "System prompt",
                value="You are an AI participating in a multi-agent discussion. "
                      "Be concise (2-4 sentences), stay on topic, and respond to the previous message.",
                height=100,
                label_visibility="collapsed",
            )

        col_chat_run, col_chat_clear = st.columns([3, 1])
        chat_run = col_chat_run.button(
            "▶ Start Conversation",
            use_container_width=True,
            type="primary",
            disabled=not chat_topic.strip(),
        )
        if col_chat_clear.button("🗑 Clear chats", use_container_width=True, key="clear_chat"):
            st.session_state.chat_logs.clear()
            st.rerun()

        # ── live conversation ──
        if chat_run and chat_topic.strip():
            st.divider()
            st.subheader("💬 Conversation")

            live_log: list[dict] = []
            chat_container = st.container()

            def on_message(msg: ChatMessage) -> None:
                live_log.append({
                    "llm_name": msg.llm_name,
                    "model": msg.model,
                    "content": msg.content,
                    "error": msg.error,
                    "turn": msg.turn,
                })
                is_human = msg.llm_name == "Human"
                with chat_container:
                    with st.chat_message("user" if is_human else "assistant",
                                          avatar="🧑" if is_human else "🤖"):
                        label = f"**{msg.llm_name}**" + (f" `{msg.model}`" if msg.model else "")
                        st.markdown(label)
                        if msg.error:
                            st.error(msg.error)
                        else:
                            st.write(msg.content)

            with st.spinner("Conversation in progress…"):
                run_llm_conversation(
                    configs=chat_configs,
                    opening_message=chat_topic.strip(),
                    num_turns=int(num_turns),
                    system_prompt=system_prompt,
                    timeout=int(chat_timeout),
                    on_message=on_message,
                )

            st.session_state.chat_logs.append(live_log)
            st.success(f"Conversation finished — {len(live_log)} messages.")

        # ── past conversations ──
        if st.session_state.chat_logs:
            past = st.session_state.chat_logs[:-1] if chat_run else st.session_state.chat_logs
            # always show latest (may already be rendered live above if just run)
            latest_log = st.session_state.chat_logs[-1]

            if not chat_run:
                st.divider()
                st.subheader("💬 Last Conversation")
                _render_chat_log(latest_log, st.session_state.llm_configs)

            if len(st.session_state.chat_logs) > 1:
                st.divider()
                st.subheader(f"🕓 Chat History ({len(st.session_state.chat_logs)} conversations)")
                for i, clog in enumerate(reversed(st.session_state.chat_logs)):
                    label = f"Conversation {len(st.session_state.chat_logs)-i}"
                    if clog:
                        # show opening message as preview
                        opening = next((m["content"] for m in clog if m["llm_name"] == "Human"), "")
                        label += f" — {opening[:60]}{'…' if len(opening) > 60 else ''}"
                    with st.expander(label, expanded=False):
                        _render_chat_log(clog, st.session_state.llm_configs)
