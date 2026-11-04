# LLM Availability Checker

Aplikasi web kecil untuk mengecek semua model dari **satu provider OpenAI-compatible**, mengukur latency, dan menampilkan nama model yang siap langsung di-copy-paste.

## Menjalankan

Membutuhkan Python 3.10 atau lebih baru. Tidak membutuhkan package tambahan.

```powershell
$env:LLM_API_KEY="sk-..."
python app.py
```

Buka `http://127.0.0.1:8000`. API key dan konfigurasi lainnya otomatis disimpan di browser (`localStorage`) sehingga tidak perlu diisi ulang terus-menerus.

## Menjalankan versi EXE

Jalankan `dist\LLMAvailabilityChecker.exe`, lalu buka `http://127.0.0.1:8000` di browser.
EXE ini tidak membutuhkan instalasi Python atau package tambahan.

## Provider lain

Isi **Base URL provider** dengan URL API yang kompatibel OpenAI, misalnya:

```text
https://api.openai.com/v1
https://api.groq.com/openai/v1
https://openrouter.ai/api/v1
```

Provider harus mendukung:

- `GET /models`
- `POST /chat/completions`

Model diuji dengan prompt singkat, `max_tokens: 16`, dan `temperature: 0`. Pengujian memakai concurrency maksimum 10 model agar tidak membebani provider. Hasil awalnya diurutkan dari latency terendah; filter latency maksimum bersifat opsional.

## Filter dan pengurutan

- **Cari model** menyaring berdasarkan nama (tidak membedakan huruf besar/kecil).
- **Hanya model chat** menyembunyikan model yang jelas bukan chat (embedding, image, audio, moderation, dan sejenisnya) agar tidak tampil sebagai FAILED.
- Klik judul kolom **Status**, **Model**, **Latency**, atau **Keterangan** untuk mengurutkan; klik lagi untuk membalik arah. Panah ▲/▼ menandai kolom dan arah aktif.
- Jumlah baris yang sedang tampil ditunjukkan di samping filter (misalnya `2 dari 7 model`).

## Export hasil

- **Export JSON** menyimpan hasil pengecekan sebagai JSON. Bila filter sedang aktif, isinya mengikuti baris yang tampil dan diberi penanda `filtered: true` serta `exported_total`.
- **Export CSV** menyimpan `status, model, latency_ms, note` sesuai baris yang tampil; nilai yang mengandung koma atau kutip di-escape otomatis.
- Nama file memakai waktu pengecekan, misalnya `llm-check-20261104070000.csv`.

## Riwayat pengecekan

Setiap pengecekan otomatis disimpan ke `localStorage` (maksimum 8 entri terakhir) beserta Base URL, ringkasan, dan status tiap model. Gunakan **Tampilkan** untuk memuat ulang hasil lama tanpa memanggil provider, **Hapus** untuk satu entri, atau **Hapus semua** untuk membersihkan riwayat. API key tidak pernah disimpan di riwayat.

## Konfigurasi environment

| Variable | Default | Keterangan |
|---|---|---|
| `LLM_API_KEY` | kosong | API key default pada form |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL provider default |
| `LLM_TEST_PROMPT` | `Reply with exactly: OK` | Prompt pengujian default |
| `HOST` | `127.0.0.1` | Host server |
| `PORT` | `8000` | Port server |
