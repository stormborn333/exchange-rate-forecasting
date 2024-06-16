FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/er_forecast.py .

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.app:server"]