FROM python:3.11-slim

WORKDIR /app

RUN pip install uv
RUN uv pip install --system \
    "torch>=2.0.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.5.0" \
    "numpy>=1.24.0" \
    "scikit-learn>=1.6.0"

COPY src/ src/
COPY code/7_streaming_app.py code/7_streaming_app.py
COPY models/ models/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-u", "code/7_streaming_app.py"]
