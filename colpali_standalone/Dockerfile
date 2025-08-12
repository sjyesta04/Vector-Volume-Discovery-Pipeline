FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "colpali_embedding_host_server_script:app", "--host", "0.0.0.0", "--port", "8000"]