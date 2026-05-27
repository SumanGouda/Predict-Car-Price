# ── Base Image ────────────────────────────────────────
FROM python:3.11-slim

# ── Set Working Directory ──────────────────────────────
WORKDIR /app

# ── Copy and Install Dependencies First ───────────────
# (copied separately so Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the Entire Project ────────────────────────────
COPY . .

# ── Expose Port ────────────────────────────────────────
EXPOSE 7860

# ── Start the App ──────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "website.app:app"]
