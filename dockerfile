FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей (нужно для некоторых ML библиотек)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

# Порт для Streamlit
EXPOSE 8501

# Порт для документации
EXPOSE 8000

# Команда запуска (по умолчанию Streamlit)
CMD ["streamlit", "run", "interface/app.py", "--server.address", "0.0.0.0"]