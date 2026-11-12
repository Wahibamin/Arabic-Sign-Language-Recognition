FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
