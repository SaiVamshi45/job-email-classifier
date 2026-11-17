FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements-runtime.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
# Trained models are NOT baked into the image (they are derived from private mail); mount them:
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models:ro" jobtracker-api
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "jobtracker.api:app", "--host", "0.0.0.0", "--port", "8000"]
