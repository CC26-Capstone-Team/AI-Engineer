# Gunakan image Python standar yang ringan
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Tentukan working directory
WORKDIR /app

# Salin file requirements
COPY ./requirements.txt /app/requirements.txt

# Install library dengan versi yang sama persis dengan Colab
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi dan model
COPY ./app /app/app
COPY ./mlp_job_title.keras /app/mlp_job_title.keras
COPY ./model_metadata.json /app/model_metadata.json
COPY ./mlb_edu_bg.pkl /app/mlb_edu_bg.pkl
COPY ./mlb_skills.pkl /app/mlb_skills.pkl
COPY ./encoder_education_required.pkl /app/encoder_education_required.pkl
COPY ./scaler_gpa.pkl /app/scaler_gpa.pkl

EXPOSE 8080

# Jalankan Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]