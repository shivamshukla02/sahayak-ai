# Use lightweight official Python runtime base image
FROM python:3.10-slim

# Prevent Python from writing pyc files to disk and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential for C++ bindings if building vector clients)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download standard sentence transformer model during image build time
# This guarantees that RAG features will work 100% offline immediately upon first boot!
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy remaining code files into container
COPY . /app/

# Expose local FastAPI port
EXPOSE 8000

# Command to run uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
