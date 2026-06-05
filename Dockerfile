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
COPY ./model /app/model
COPY ./encoder /app/encoder

EXPOSE 8080

# Jalankan Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]