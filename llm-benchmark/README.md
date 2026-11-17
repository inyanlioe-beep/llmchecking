# LLM Benchmark

Aplikasi benchmark untuk menguji 2+ LLM secara paralel.

## Fitur
- Uji beberapa LLM secara paralel
- Ukur latency (TTFT & total), tokens/s, throughput
- Penilaian kualitas otomatis (relevance via judge LLM, opsional)
- Konfigurasi endpoint OpenAI-compatible (URL + API key + model name)
- Riwayat hasil benchmark
- Ekspor CSV

## Setup

```bash
cd llm-benchmark
pip install -r requirements.txt
streamlit run app.py
```

## Konfigurasi LLM

Tambah provider di UI: masukkan **Base URL**, **API Key**, dan **Model name**.

Contoh endpoint OpenAI-compatible:
- OpenAI: `https://api.openai.com/v1`
- Groq: `https://api.groq.com/openai/v1`
- Ollama: `http://localhost:11434/v1`
- LM Studio: `http://localhost:1234/v1`
- Custom proxy: URL sesuai server

## Metrik
| Metrik | Keterangan |
|---|---|
| TTFT | Time to First Token (ms) |
| Total latency | Waktu hingga respons selesai (s) |
| Tokens/s | Throughput output token |
| Total tokens | Input + output tokens |
| Quality score | Skor relevansi 1-10 (via judge, opsional) |
