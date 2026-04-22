FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]