FROM almalinux:9

# Install Python 3.11 and create proper symlinks
RUN dnf install -y python3.11 python3.11-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies using dnf
RUN dnf install -y \
    glib2 \
    libSM \
    libXext \
    libXrender \
    libgomp \
    mesa-libGL \
    libjpeg-turbo \
    libpng \
    curl \
    && dnf clean all

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Create necessary directories and set proper ownership
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app/logs

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]