#!/bin/bash
# Запуск всей инфраструктуры AutoML (приложение + MLFlow + PostgreSQL)

set -e

echo "Запуск AutoML инфраструктуры..."
echo ""

# Переход в директорию проекта
cd "$(dirname "$0")/.."

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "Docker не найден. Установите Docker Desktop."
    exit 1
fi

# Проверка docker compose
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif docker-compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo "Docker Compose не найден."
    exit 1
fi

echo "Запуск сервисов через $COMPOSE_CMD..."
$COMPOSE_CMD up -d --build

echo ""
echo "Ожидание запуска сервисов (30 секунд)..."
sleep 30

echo ""
echo "Инфраструктура запущена!"
echo ""
echo "Сервисы доступны по адресам:"
echo "AutoML UI:      http://localhost:8501"
echo "Documentation:  http://localhost:8000"
echo "MLFlow UI:      http://localhost:5050"
echo "PostgreSQL:    localhost:5432"
echo ""
echo "Полезные команды:"
echo "Логи MLFlow:     $COMPOSE_CMD logs -f mlflow-service"
echo "Логи AutoML:     $COMPOSE_CMD logs -f automl"
echo "Остановка:       $COMPOSE_CMD down"
echo "Полная очистка:  $COMPOSE_CMD down -v"
echo ""
