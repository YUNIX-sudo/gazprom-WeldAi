FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY frontend/ ./frontend

EXPOSE 8000

COPY model.pt /app/model.pt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
