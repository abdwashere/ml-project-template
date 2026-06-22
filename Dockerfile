# ============================================================
# This Dockerfile separates heavy/stable dependencies from
# light/frequently-changed ones, so editing requirements.txt
# doesn't force a full reinstall of everything (e.g. torch).
#
# If your project uses torch/tensorflow, add it to the
# "heavy deps" layer below — NOT to requirements.txt — so it
# stays cached across builds.
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# --- Heavy, stable dependencies (rarely change, cached separately) ---
# Example, uncomment if needed:
# RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# --- Light, frequently-changed dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ---
COPY src/ ./src/
COPY api/ ./api/
COPY monitoring/ ./monitoring/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]