FROM python:3.12-slim

WORKDIR /app

# 使用国内镜像源加速下载
RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://security.debian.org/debian-security|http://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 使用国内 pip 镜像
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml ./

# 安装 PyTorch CPU 版 + 全部依赖（一条命令，避免 CUDA 版覆盖）
# --extra-index-url 让 torch 从 CPU wheel 源下载（~180MB vs ~2.5GB CUDA 版）
RUN uv pip install --system --no-cache-dir \
    --index-strategy unsafe-best-match \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    .

# Copy backend code
COPY backend/ ./backend/

# Copy data directory (golden dataset, etc.)
COPY data/eval/ ./data/eval/

# Create data directories that will be mounted at runtime
RUN mkdir -p /app/data/documents /app/data/eval/reports /app/backend/uploads

# Set working directory to backend
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
