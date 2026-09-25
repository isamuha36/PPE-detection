# Real-Time PPE Compliance Detection in Mining Environments
### Proyek Tugas Akhir - D4 Teknologi Rekayasa Perangkat Lunak (TRPL) SV UGM
**Mitra:** PT Parama Data Unit  
**Fokus Bidang:** AI & Data Science (Model Engineering, Inference Optimization & Edge Deployment)

---

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem deteksi kepatuhan **Alat Pelindung Diri (APD)** secara *real-time* pada lingkungan pertambangan menggunakan *deep learning object detection*. Sistem dirancang untuk mengidentifikasi keberadaan serta kepatuhan pekerja tambang dalam mengenakan APD standar K3 (Kesehatan dan Keselamatan Kerja).

### Kelas Deteksi Target
1. **Cap / Helmet**: `cap_on` (memakai helm) vs `cap_off` (tidak memakai helm)
2. **Safety Mask**: `mask_on` (memakai masker) vs `mask_off` (tidak memakai masker)
3. **Safety Jacket / Hi-Vis Vest**: `jacket_on` (memakai rompi) vs `jacket_off` (tidak memakai rompi)
4. **Safety Gloves**: `gloves_on` (memakai sarung tangan) vs `gloves_off` (tidak memakai sarung tangan)
5. **Person Anchor**: `person` (bounding box pekerja)

---

## 2. Struktur Repositori
```text
PPE-detection/
├── configs/               # Konfigurasi dataset (data.yaml) & hyperparameter
├── data/
│   ├── raw/               # Dataset mentah / arsip unduhan
│   ├── processed/         # Dataset berformat YOLO (train, val, test)
│   └── samples/           # Sampel gambar/video pengujian inferensi
├── docs/                  # Dokumentasi teknis & pedoman laporan
├── models/
│   ├── pretrained/        # Weights pretrained awal (yolov8n.pt / yolo11s.pt)
│   ├── trained/           # Checkpoints hasil training (best.pt)
│   └── exported/          # Model teroptimasi (ONNX, TensorRT engine)
├── notebooks/             # Eksplorasi data, augmentasi, & visualisasi
├── scripts/               # CLI scripts (train, evaluate, export, benchmark)
├── src/
│   ├── dataset/           # Data loading, conversion, & augmentation
│   ├── training/          # Training loop & callbacks
│   ├── optimization/      # Model quantization & export engine (ONNX/TRT)
│   ├── inference/         # Detector engine (PyTorch, ONNX, TensorRT)
│   └── service/           # FastAPI backend & CCTV stream worker
├── tests/                 # Unit testing & automated validation
├── requirements.txt       # Daftar dependensi Python
└── README.md
```

---

## 3. Aspek Engineering & Evaluasi (Standar TRPL SV UGM)
1. **Model Evaluation**:
   * Evaluasi akurasi deteksi: *Precision*, *Recall*, *mAP@50*, *mAP@50-95*, *F1-score*, *Confusion Matrix*.
2. **Inference Optimization (Baseline vs Optimized)**:
   * **Baseline**: PyTorch standard FP32 (`.pt`).
   * **Optimized**: ONNX Runtime GPU (FP16) & TensorRT engine (`.engine`).
   * **Metrik Evaluasi Teknis**: *Inference Latency (ms)*, *Throughput (FPS)*, *VRAM / Memory Consumption*.
3. **Deployment / Integration**:
   * *Inference Service* berbasis FastAPI / OpenCV pipeline untuk memproses feed video/CCTV dan mencatat log pelanggaran secara otomatis.
