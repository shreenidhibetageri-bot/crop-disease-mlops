FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install numpy==1.23.5 && \
    pip install tensorflow==2.12.0 && \
    pip install fastapi==0.95.2 \
                uvicorn==0.22.0 \
                python-multipart \
                pillow requests \
                scikit-learn \
                matplotlib \
                seaborn pandas \
                mlflow \
                pydantic==1.10.13

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]