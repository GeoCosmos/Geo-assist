# Image for the containerised deployment (see deploy/docker-compose.yml).
#
# Kept in version control because it was previously untracked and lived only on
# the deployment VM, which made every change a hand-edit over SSH.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ocr.txt

# Bake the easyocr models into the image. Left to first use they download into
# ~/.EasyOCR inside the container, which is ephemeral — so they refetch on every
# restart and the runtime needs internet access to work at all.
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . .

EXPOSE 8743

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8743"]
