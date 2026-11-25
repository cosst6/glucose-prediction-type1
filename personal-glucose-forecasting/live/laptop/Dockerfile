FROM python:3.13-slim
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY . .
RUN useradd -m app && chown -R app /app
USER app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
EXPOSE 7860
CMD ["python", "server.py"]
