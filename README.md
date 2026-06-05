# 🤖 AI Engineer - Job Title Prediction Model & API



Selamat datang di repositori layanan kecerdasan buatan (*AI Service*) untuk **Job Title Prediction**. Repositori ini berisi pipa pemodelan mendalam (*Deep Learning*) menggunakan **TensorFlow Functional API** hingga proses *deployment* siap pakai sebagai REST API menggunakan **FastAPI** di lingkungan **Google Cloud Run**.



Sistem ini dirancang untuk menerima input profil kandidat mentah yang mudah dipahami manusia (*Human-readable input*), melakukan transformasi data secara *real-time* di memori melalui berkas preprocessor (`.pkl`), dan memprediksi **20 kelas Job Title** terbaik lengkap dengan nilai probabilitasnya (*confidence score*).



---### 📂 Struktur Direktori Proyek

Proyek ini diorganisasikan secara modular untuk memisahkan antara proses riset, penyimpanan artefak biner, dan kode produksi aplikasi:```text

job-title-prediction/

├── app/

│   └── main.py                 # Kode utama FastAPI (Inference Service)

├── encoder/

│   ├── encoder_education_required.pkl # Label encoder tingkat pendidikan

│   ├── mlb_edu_bg.pkl          # Multi-label binarizer rumpun studi

│   ├── mlb_skills.pkl          # Multi-label binarizer keahlian kandidat

│   └── scaler_gpa.pkl          # Scaler normalisasi nilai GPA (IPK)

├── model/

│   ├── mlp_job_title.keras     # Bobot & Graf Model Utama (Dieksklusi dari Git)

│   └── model_metadata.json     # Konfigurasi target kelas & info akurasi

├── carpathmu_mlp_modelling.ipynb # Jupyter Notebook training & eksperimen

├── Dockerfile                  # Cetak biru Docker untuk standardisasi deployment

├── requirements.txt            # Daftar dependensi library python proyek

└── .gitignore                  # Berkas pengecualian pelacakan Git (.keras diabaikan)

🧠 Tentang Model (Deep Learning Architecture)

Seluruh proses riset, eksplorasi, dan pelatihan model didokumentasikan pada berkas carpathmu_mlp_modelling.ipynb. Spesifikasi utama model ini adalah:

Arsitektur Jaringan: Menggunakan Multi-Layer Perceptron (MLP) tingkat dalam yang memanfaatkan struktur Residual Connection (Skip Connection) untuk memitigasi risiko vanishing gradient pada jaringan dalam.

Komponen Kustom (Custom Components):

ResidualBlock: Blok kustom berbasis objek layers.Layer yang menggabungkan lapisan Dense, Batch Normalization, Activation (ReLU), dan Dropout (rate: 0.3) dengan proyeksi linear jalan pintas (skip layer).

LabelSmoothingCategoricalCrossentropy: Fungsi kerugian kustom berbasis objek losses.Loss dengan teknik label smoothing sebesar 0.1 guna mencegah model menjadi terlalu percaya diri (overconfident) akibat distribusi kelas yang tidak seimbang.

Fitur Input: Model mengekspektasikan total 104 fitur hasil penggabungan matriks biner dan numerik secara horizontal (axis=1) sesaat sebelum proses inference.

⚡ Dokumentasi & Integrasi FastAPI (Cloud Run Deployment)

Layanan ini dikemas menggunakan FastAPI demi performa tinggi dan latensi rendah saat melayani permintaan prediksi dari aplikasi Front-End atau Mobile.

🌐 URL Live Production & Dokumentasi

Production API Base URL: https://<nama-service-kamu>-<hash-cloud-run>.a.run.app

Live Interactive Swagger UI: https://<nama-service-kamu>-<hash-cloud-run>.a.run.app/docs

Live Alternative ReDoc: https://<nama-service-kamu>-<hash-cloud-run>.a.run.app/redoc

Service Health Check: https://<nama-service-kamu>-<hash-cloud-run>.a.run.app/health

💡 Silakan sesuaikan tautan di atas dengan URL resmi yang didapatkan dari Google Cloud Run Console setelah proses deploy selesai.

1. Cara Menjalankan API Secara Lokal (Tahap Pengembangan)

Jika ingin menguji API atau melakukan perubahan kode secara lokal di komputer Anda:

Bash



# Buka terminal di direktori root proyek (job-title-prediction)# 1. Install seluruh dependensi library Python

pip install -r requirements.txt# 2. Jalankan server lokal Uvicorn dengan auto-reload

uvicorn app.main:app --reload

Akses Swagger UI lokal tersedia melalui alamat: http://127.0.0.1:8000/docs

2. Spesifikasi Endpoint Prediksi

POST /predict

Menerima profil mentah kandidat, memprosesnya lewat pipa enkoder, dan mengembalikan hasil prediksi Top-K.

Request Body (JSON):

JSON



{

  "education_required": "Bachelor's Degree",

  "edu_bg": ["Computer Science", "Information Technology"],

  "skills": ["Python", "TensorFlow", "FastAPI", "SQL"],

  "gpa": 3.85,

  "top_k": 3

}

Response Success (JSON - Status 200):

JSON



{

  "predicted": "Machine Learning Engineer",

  "confidence": 0.8942,

  "top_k": [

    {

      "rank": 1,

      "job_title": "Machine Learning Engineer",

      "probability": 0.8942

    },

    {

      "rank": 2,

      "job_title": "Data Scientist",

      "probability": 0.0815

    },

    {

      "rank": 3,

      "job_title": "Software Engineer",

      "probability": 0.0243

    }

  ]

}

🐳 Panduan Kontainerisasi & Deployment (Google Cloud Run)

Proyek ini menggunakan Dockerfile berbasis python:3.12-slim untuk menjamin ukuran image yang ringan namun tetap stabil. Port internal container diatur pada port 8080 sesuai dengan standar mutlak infrastruktur Google Cloud.

Langkah cepat untuk membangun dan merilis aplikasi via Google Cloud CLI (gcloud):

Bash



# 1. Tentukan Project ID target di Google Cloud

gcloud config set project <PROJECT_ID_KAMU># 2. Build image dan simpan ke Google Artifact Registry / Container Registry

gcloud builds submit --tag gcr.io/<PROJECT_ID_KAMU>/job-title-api# 3. Deploy langsung image ke Google Cloud Run

gcloud run deploy job-title-service \

    --image gcr.io/<PROJECT_ID_KAMU>/job-title-api \

    --platform managed \

    --region asia-southeast2 \

    --allow-unauthenticated

