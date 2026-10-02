# Panduan Kurasi Manual Ground Truth Rig PDU (100 Frame Kunci)

Dokumen ini adalah protokol anotasi baku (*Annotation Codebook*) untuk menghasilkan dataset standar emas (*Gold Standard Ground Truth*) dari rekaman CCTV rig pengeboran PT Parama Data Unit.

---

## 1. Struktur Folder Kerja

Dataset 100 frame kunci tersimpan di:
```text
E:\Tugas Akhir\PPE-detection\data\pdu_ground_truth\
├── images/            <-- 100 citra resolusi asli 2K QHD (2560x1440) yang tersebar merata
├── labels/            <-- Anotasi draf awal dari YOLO11s (format .txt koordinat YOLO)
├── visualized/        <-- Preview visual kotak deteksi awal untuk memudahkan pengecekan
├── classes.txt        <-- Daftar 9 kelas baku standar
└── GUIDE_KURASI.md    <-- [Dokumen Ini] Panduan langkah demi langkah
```

---

## 2. Protokol Anotasi Objek APD Rig PDU (*Annotation Codebook*)

Gunakan aturan berikut agar kualitas label konsisten dan ilmiah:

| Class ID | Nama Kelas | Objek Nyata di Rig PDU | Batasan Kotak Bounding Box |
| :---: | :--- | :--- | :--- |
| `0` | `person` | Driller / operator konsol rig | Kotak melingkupi seluruh tubuh pekerja yang tampak di kamera (dari atas helm hingga batas pinggang/kaki yang terlihat, meski terpotong meja konsol). |
| `1` | `cap_on` | Helm keselamatan putih / kuning | Kotak pas mengelilingi batas fisik helm yang dipakai pekerja. |
| `2` | `cap_off` | Kepala pekerja tanpa helm | Kotak pas mengelilingi kepala / rambut jika ada pekerja yang melepas helm. |
| `3` | `mask_on` | Masker debu / respirator | Kotak pas mengelilingi masker jika pekerja memakainya. |
| `4` | `mask_off` | Wajah tanpa masker | Wajah pekerja tanpa masker. |
| `5` | `jacket_on` | Wearpack coverall oranye | Kotak mengelilingi bagian pakaian wearpack terusan oranye (dari leher/bahu hingga batas baju terbawah yang terlihat). |
| `6` | `jacket_off`| Pekerja bertelanjang dada / tanpa seragam | Ditandai jika pekerja tidak memakai wearpack kerja. |
| `7` | `gloves_on` | Sarung tangan kerja | Kotak pas mengelilingi pergelangan hingga ujung jari tangan yang bersarung tangan. |
| `8` | `gloves_off`| Tangan telanjang tanpa sarung tangan | Kotak mengelilingi telapak / jari tangan yang terbuka tanpa sarung tangan. |

### Aturan Khusus Pembersihan Anomali (*False Positive Clean-Up*):
* **HAPUS** kotak apapun yang menempel pada manometer / *pressure gauge* analog di meja konsol.
* **HAPUS** kotak apapun yang menempel pada pipa hidrolik merah atau selang kabel.
* **HAPUS** kotak apapun yang menempel pada rangka baja atau komponen mesin derek kuning.
* **HAPUS** kotak helm yang menempel pada sandaran kursi operator.

---

## 3. Pilihan Aplikasi Anotasi & Cara Penggunaannya

### Opsi A: Menggunakan Roboflow (Sangat Direkomendasikan - Paling Cepat & Gratis)
1. Buka browser dan kunjungi `https://roboflow.com` (buat akun gratis jika belum ada).
2. Klik **Create New Project**:
   * Project Type: **Object Detection**
   * Annotation Group: Masukkan nama kelas dari `classes.txt`.
3. Klik **Upload Data**:
   * Tarik (*drag-and-drop*) seluruh isi folder `data/pdu_ground_truth/images/` dan `data/pdu_ground_truth/labels/`.
   * Roboflow akan otomatis membaca gambar beserta kotak draf awal yang sudah kita siapkan.
4. Lakukan **Review & Editing**:
   * Klik kotak yang salah (seperti di manometer / pipa) -> tekan tombol `Delete` di keyboard.
   * Jika ada wearpack oranye pekerja yang belum terkotaki, tekan tombol `W` lalu tarik kotak baru dan pilih `jacket_on`.
5. Selesai & Ekspor:
   * Klik **Generate Dataset** -> pilih format **YOLOv11** (atau YOLOv8).
   * Unduh file `.zip`, ekstrak label yang sudah bersih kembali ke proyek kita.

### Opsi B: Menggunakan Label Studio (Lokal di Laptop / 100% Offline)
Jika data tidak boleh diunggah ke internet karena kebijakan kerahasiaan data industri:
1. Buka terminal di folder proyek:
   ```bash
   .venv\Scripts\pip install label-studio
   .venv\Scripts\label-studio start
   ```
2. Buka `http://localhost:8080` di browser.
3. Buat project baru -> Pilih template **Object Detection with Bounding Boxes**.
4. Masukkan kelas-kelas APD dari `classes.txt`.
5. Import data lokal dari `data/pdu_ground_truth/images/`.

---

## 4. Alur Setelah 100 Frame Bersih Tersedia

Setelah 100 frame ini selesai dirapikan:
1. Kita akan membagi dataset menjadi:
   * **75 frame** untuk *Fine-Tuning Latih (Train Set)*.
   * **25 frame** untuk *Evaluasi Objektif (Validation & Benchmark Set)*.
2. Kita jalankan proses transfer learning fine-tuning pada model YOLO11s.
3. Model baru akan memiliki akurasi di atas 85–90% pada CCTV rig PDU tanpa lagi mengalami salah deteksi pada mesin atau manometer.
