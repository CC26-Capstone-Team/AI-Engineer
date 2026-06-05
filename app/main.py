import json
import os
from typing import List
from fastapi import FastAPI, HTTPException
import numpy as np
import joblib  # Wajib untuk memuat file .pkl
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

app = FastAPI(
    title="Job Title Prediction API",
    description="API untuk memprediksi 20 kelas job title berdasarkan fitur input asli.",
    version="2.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. REDEFINISI CUSTOM COMPONENTS (Perbaikan bug deserialisasi Keras) ──
class ResidualBlock(layers.Layer):
    def __init__(self, units: int, dropout_rate: float = 0.3, l2: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1 = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation("relu")
        self.drop1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))
        self.bn2 = layers.BatchNormalization()
        self.projection = layers.Dense(units, use_bias=False)
        self.bn_proj = layers.BatchNormalization()
        self.act_out = layers.Activation("relu")

    def call(self, x, training=False):
        h = self.dense1(x)
        h = self.bn1(h, training=training)
        h = self.act1(h)
        h = self.drop1(h, training=training)
        h = self.dense2(h)
        h = self.bn2(h, training=training)
        skip = self.projection(x)
        skip = self.bn_proj(skip, training=training)
        return self.act_out(h + skip)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return cfg


class LabelSmoothingCategoricalCrossentropy(keras.losses.Loss):
    # Ditambahkan **kwargs untuk menangkap argumen otomatis 'reduction' dari Keras saat load_model
    def __init__(self, num_classes: int, smoothing: float = 0.1, name: str = "label_smoothing_cce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), self.num_classes)
        y_smooth = y_true_oh * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        return -tf.reduce_mean(tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "smoothing": self.smoothing})
        return cfg


# ── 2. PATH ABSOLUT ARTEFAK DI DOCKER CONTAINER ──
MODEL_PATH = "/app/mlp_job_title.keras"
METADATA_PATH = "/app/model_metadata.json"

# Definisikan path untuk file pkl preprocessing Anda
ENC_EDU_REQ_PATH = "/app/encoder_education_required.pkl"
MLB_EDU_BG_PATH = "/app/mlb_edu_bg.pkl"
MLB_SKILLS_PATH = "/app/mlb_skills.pkl"
SCALER_GPA_PATH = "/app/scaler_gpa.pkl" 

# Inisialisasi variabel global kosong
MODEL = None
META = {}
CLASS_NAMES = []
STARTUP_ERROR = None

# Variabel objek encoder/scaler global
ENC_EDU_REQ = None
MLB_EDU_BG = None
MLB_SKILLS = None
SCALER_GPA = None

try:
    print(f"Mencoba memuat metadata dari: {METADATA_PATH}")
    with open(METADATA_PATH, "r") as f:
        META = json.load(f)
    CLASS_NAMES = META["class_names"]

    print(f"Mencoba memuat model Keras dari: {MODEL_PATH}")
    MODEL = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "ResidualBlock": ResidualBlock,
            "LabelSmoothingCategoricalCrossentropy": LabelSmoothingCategoricalCrossentropy,
        },
        compile=False
    )
    
    print("Mencoba memuat file-file preprocessing .pkl...")
    ENC_EDU_REQ = joblib.load(ENC_EDU_REQ_PATH)
    MLB_EDU_BG = joblib.load(MLB_EDU_BG_PATH)
    MLB_SKILLS = joblib.load(MLB_SKILLS_PATH)
    
    if os.path.exists(SCALER_GPA_PATH):
        SCALER_GPA = joblib.load(SCALER_GPA_PATH)
        print("Scaler GPA pkl berhasil dimuat.")

    print("Model, Metadata, dan seluruh Preprocessor pkl berhasil dimuat!")
except Exception as e:
    STARTUP_ERROR = str(e)
    print(f"CRITICAL STARTUP ERROR: {STARTUP_ERROR}")


# ── 3. SKEMA PYDANTIC BARU (MENERIMA FITUR MANUSIA / MENTAH) ──
class PredictRequest(BaseModel):
    education_required: str     # Contoh: "Bachelor's Degree"
    edu_bg: List[str]            # Contoh: ["Computer Science", "Information Technology"]
    skills: List[str]            # Contoh: ["Python", "TensorFlow", "SQL"]
    gpa: float                   # Contoh: 3.75
    top_k: int = 3

class PredictionItem(BaseModel):
    rank: int
    job_title: str
    probability: float

class PredictResponse(BaseModel):
    predicted: str
    confidence: float
    top_k: List[PredictionItem]


# ── 4. ENDPOINT ROUTES ──
@app.get("/health")
def health_check():
    if STARTUP_ERROR:
        return {
            "status": "unhealthy",
            "error": STARTUP_ERROR,
            "message": "Model atau preprocessor gagal dimuat. Periksa log Docker container Anda."
        }
    return {
        "status": "healthy",
        "model_name": META.get("model_name", "Unknown"),
        "accuracy": META.get("best_val_acc", 0.0),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Proteksi jika model atau pkl gagal dimuat saat startup
    if MODEL is None or ENC_EDU_REQ is None or MLB_EDU_BG is None or MLB_SKILLS is None:
        raise HTTPException(
            status_code=503,
            detail=f"Layanan tidak siap. Artefak gagal dimuat pada startup server. Error: {STARTUP_ERROR}"
        )

    try:
        # ── PROSES 1: Preprocessing 'education_required' ──
        # Menghasilkan array angka berdasarkan Label/Ordinal Encoder Anda
        edu_req_encoded = ENC_EDU_REQ.transform([[req.education_required]])
        edu_req_feat = np.array(edu_req_encoded, dtype=np.float32).reshape(1, -1)

        # ── PROSES 2: Preprocessing 'edu_bg' ──
        # Mengubah list background pendidikan menjadi biner multi-kolom via MultiLabelBinarizer
        edu_bg_feat = MLB_EDU_BG.transform([req.edu_bg]).astype(np.float32)

        # ── PROSES 3: Preprocessing 'skills' ──
        # Mengubah list teks skill menjadi biner multi-kolom via MultiLabelBinarizer
        skills_feat = MLB_SKILLS.transform([req.skills]).astype(np.float32)

        # ── PROSES 4: Preprocessing 'gpa' ──
        if SCALER_GPA:
            gpa_scaled = SCALER_GPA.transform([[req.gpa]])[0][0]
        else:
            # Fallback pembagian manual jika scaler .pkl tidak diexport dari notebook
            gpa_scaled = req.gpa / 4.0 
        gpa_feat = np.array([[gpa_scaled]], dtype=np.float32)

        # ── PROSES 5: MENGGABUNGKAN SELURUH FITUR SECARA HORIZONTAL (axis=1) ──
        # Urutan penggabungan ini menghasilkan total 104 fitur sesuai spesifikasi model Keras Anda
        x = np.concatenate([edu_req_feat, edu_bg_feat, skills_feat, gpa_feat], axis=1)
        
        # Validasi dimensi kecocokan fitur input
        expected_features = META.get("num_features", 104)
        if x.shape[1] != expected_features:
            raise ValueError(
                f"Kombinasi data menghasilkan {x.shape[1]} fitur, sedangkan model mengekspektasikan {expected_features} fitur."
            )

        # ── PROSES 6: EXECUTE FORWARD PASS / INFERENCE ──
        probs = MODEL.predict(x, verbose=0)[0]
        
        # Sorting nilai probabilitas tertinggi
        top_indices = np.argsort(probs)[::-1][: req.top_k]

        prediction_items = [
            PredictionItem(rank=r + 1, job_title=CLASS_NAMES[idx], probability=float(probs[idx]))
            for r, idx in enumerate(top_indices)
        ]

        return PredictResponse(
            predicted=CLASS_NAMES[top_indices[0]],
            confidence=float(probs[top_indices[0]]),
            top_k=prediction_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan preprocessing atau inference: {str(e)}")