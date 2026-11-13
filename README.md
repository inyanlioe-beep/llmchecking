# LLM Availability Checker

Aplikasi web kecil untuk mengecek semua model dari **satu provider OpenAI-compatible**, mengukur latency, dan menampilkan nama model yang siap langsung di-copy-paste.

## Menjalankan

Membutuhkan Python 3.10 atau lebih baru. Tidak membutuhkan package tambahan.

```powershell
$env:LLM_API_KEY="sk-..."
python app.py
```

Buka `http://127.0.0.1:8000`. API key dan konfigurasi lainnya otomatis disimpan di browser (`localStorage`) sehingga tidak perlu diisi ulang terus-menerus.

## Endpoint tersimpan

1. Isi **Base URL provider**, **API key**, dan **Nama endpoint** (opsional; default nama host provider).
2. Klik **Simpan endpoint** untuk menyimpan pasangan Base URL dan API key.
3. Untuk pengecekan berikutnya, pilih profil di **Endpoint tersimpan**, lalu klik **Cek semua model**. Base URL dan API key akan terisi otomatis; konfigurasi terakhir juga dimuat kembali setelah halaman dibuka ulang.
4. Untuk menambah profil lain, klik **Endpoint baru**. Untuk mengubah profil terpilih, edit isinya lalu klik **Perbarui endpoint**. **Hapus endpoint** menghapus profil terpilih dan mengosongkan API key pada form.

Profil disimpan di `localStorage` browser, termasuk API key tanpa enkripsi. Gunakan browser dan alamat aplikasi yang sama (termasuk port) untuk mengakses profil tersimpan. API key tidak dimasukkan ke riwayat pengecekan.

## Menjalankan versi EXE

Jalankan `dist\LLMAvailabilityChecker.exe`, lalu buka `http://127.0.0.1:8000` di browser. Versi ini menyertakan fitur endpoint tersimpan. Tutup aplikasi yang sedang berjalan sebelum mengganti EXE dengan versi baru, lalu jalankan kembali.
EXE ini tidak membutuhkan instalasi Python atau package tambahan. API hanya menerima request dari localhost; Base URL HTTP hanya diizinkan untuk localhost dan tujuan privat ditolak.

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

Tombol **Batalkan** menghentikan permintaan aktif dan membatalkan tes model yang belum berjalan; request jaringan yang sedang berlangsung tetap berhenti setelah timeout. Model diuji dengan prompt singkat, `max_tokens: 16`, dan `temperature: 0`. Pengujian memakai concurrency maksimum 10 model agar tidak membebani provider. Hasil awalnya diurutkan dari latency terendah; filter latency maksimum bersifat opsional. Gunakan **Filter nama model** untuk membatasi model yang diuji, dan **Maksimum model** untuk membatasi jumlah model (default 200, maksimum 500).

## Filter dan pengurutan

- **Cari model** menyaring berdasarkan nama (tidak membedakan huruf besar/kecil).
- **Hanya model chat** menyembunyikan model yang jelas bukan chat (embedding, image, audio, moderation, dan sejenisnya) agar tidak tampil sebagai FAILED.
- Klik judul kolom **Status**, **Model**, **Latency**, atau **Keterangan** untuk mengurutkan; klik lagi untuk membalik arah. Panah ▲/▼ menandai kolom dan arah aktif.
- Jumlah baris yang sedang tampil ditunjukkan di samping filter (misalnya `2 dari 7 model`).

## Tes otomatis

Jalankan tes inti dengan:

```powershell
python -m unittest -v test_app
```

Tes profil endpoint di antarmuka dapat dijalankan dengan Node.js tanpa package tambahan:

```powershell
node --test test_profiles.cjs
```

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
