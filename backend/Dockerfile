FROM python:3.14-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for background workers.
RUN useradd \
    --create-home \
    --shell /bin/bash \
    appuser

COPY app ./app

# Give the application user ownership of the application directory.
RUN chown -R appuser:appuser /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]