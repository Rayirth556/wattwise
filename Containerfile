# NOTE: --server.address=:: binds dual-stack (IPv4 + IPv6) so Windows "localhost" (::1) works through WSL2.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 WATTWISE_DATA_DIR=/app/data
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code + trained model/metadata (dataset is mounted at run time, not baked in)
COPY .streamlit ./.streamlit
COPY app ./app
COPY model/artifacts ./model/artifacts

RUN useradd --create-home --uid 1000 wattwise && chown -R wattwise /app
USER wattwise

EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=::", "--server.port=8501"]
