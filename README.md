# 🤖 Job Title Prediction API

Layanan AI berbasis Deep Learning untuk memprediksi Job Title kandidat berdasarkan latar belakang pendidikan, keahlian, dan nilai GPA.

Model dikembangkan menggunakan TensorFlow Functional API dan dideploy sebagai REST API menggunakan FastAPI pada Google Cloud Run.

## 🌐 Live Demo

| Service | URL |
|----------|-----|
| API Base URL | https://job-title-api-842754562942.asia-southeast2.run.app |
| Swagger Docs | https://job-title-api-842754562942.asia-southeast2.run.app/docs |
| ReDoc | https://job-title-api-842754562942.asia-southeast2.run.app/redoc |
| Health Check | https://job-title-api-842754562942.asia-southeast2.run.app/health |

---

## ✨ Features

* Predict **20 Job Title Classes**
* Return prediction confidence score
* Support **Top-K Prediction**
* Real-time preprocessing using serialized encoders (`.pkl`)
* REST API with interactive Swagger documentation
* Ready for containerized deployment using Docker & Cloud Run

---

## 📂 Project Structure

```text
job-title-prediction/
├── app/
│   └── main.py
│
├── encoder/
│   ├── encoder_education_required.pkl
│   ├── mlb_edu_bg.pkl
│   ├── mlb_skills.pkl
│   └── scaler_gpa.pkl
│
├── model/
│   ├── mlp_job_title.keras
│   └── model_metadata.json
│
├── carpathmu_mlp_modelling.ipynb
├── Dockerfile
├── requirements.txt
└── .gitignore
```

### Directory Description

| Directory                       | Description                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| `app/`                          | FastAPI application source code                                |
| `encoder/`                      | Preprocessing artifacts (Encoder, Scaler, MultiLabelBinarizer) |
| `model/`                        | Trained model and metadata                                     |
| `carpathmu_mlp_modelling.ipynb` | Model training & experimentation notebook                      |
| `Dockerfile`                    | Container build configuration                                  |
| `requirements.txt`              | Python dependencies                                            |

---

# 🧠 Model Architecture

The prediction model uses a **Multi-Layer Perceptron (MLP)** with **Residual Connections (Skip Connections)** to improve training stability and reduce vanishing gradient problems.

### Main Components

#### ResidualBlock

Custom TensorFlow layer consisting of:

* Dense Layer
* Batch Normalization
* ReLU Activation
* Dropout (0.3)
* Skip Connection Projection

#### LabelSmoothingCategoricalCrossentropy

Custom loss function with:

```text
Label Smoothing = 0.1
```

Used to reduce model overconfidence and improve generalization on imbalanced classes.

### Input Features

The model expects:

```text
104 Features
```

Generated from a combination of:

* Education Level Encoding
* Education Background Encoding
* Skills Encoding
* GPA Normalization

---

# 🚀 API Documentation

## Base URLs

Replace the following URL with your Cloud Run endpoint after deployment.

```text
https://<service-name>-<hash>.a.run.app
```

### Available Endpoints

| Endpoint   | Method | Description                 |
| ---------- | ------ | --------------------------- |
| `/health`  | GET    | Service health check        |
| `/predict` | POST   | Predict candidate job title |
| `/docs`    | GET    | Swagger UI                  |
| `/redoc`   | GET    | ReDoc documentation         |

---

# 💻 Running Locally

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Start Development Server

```bash
uvicorn app.main:app --reload
```

Server will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 🔮 Prediction Endpoint

## Request

### POST `/predict`

```json
{
  "education_required": "Bachelor's Degree",
  "edu_bg": [
    "Computer Science",
    "Information Technology"
  ],
  "skills": [
    "Python",
    "TensorFlow",
    "FastAPI",
    "SQL"
  ],
  "gpa": 3.85,
  "top_k": 3
}
```

---

## Response

```json
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
```

---

# 🐳 Docker Deployment

## Build Container

```bash
docker build -t job-title-api .
```

## Run Locally

```bash
docker run -p 8080:8080 job-title-api
```

---

# ☁️ Deploy to Google Cloud Run

## Set Active Project

```bash
gcloud config set project <PROJECT_ID>
```

## Build Container

```bash
gcloud builds submit \
  --tag gcr.io/<PROJECT_ID>/job-title-api
```

## Deploy Service

```bash
gcloud run deploy job-title-service \
  --image gcr.io/<PROJECT_ID>/job-title-api \
  --platform managed \
  --region asia-southeast2 \
  --allow-unauthenticated
```

After deployment, Cloud Run will generate a public URL such as:

```text
https://job-title-service-xxxxx.a.run.app
```

---

# 📊 Model Information

| Item           | Value                      |
| -------------- | -------------------------- |
| Framework      | TensorFlow                 |
| Architecture   | Deep Residual MLP          |
| Classes        | 20 Job Titles              |
| API Framework  | FastAPI                    |
| Deployment     | Google Cloud Run           |
| Container      | Docker                     |
| Input Features | 104i                       |
| Output         | Top-K Job Title Prediction |

---

## 👨‍💻 Author

Developed as part of an AI Engineering for CarPathMu Job Title Prediction and Career Recommendation Systems.
