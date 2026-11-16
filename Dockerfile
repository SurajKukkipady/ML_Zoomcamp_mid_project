FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python train.py --cv 3 --n-jobs 2

ENV MODEL_PATH=/app/models/netflix_type.joblib \
    PORT=9696
EXPOSE 9696

CMD ["python", "serve.py"]
