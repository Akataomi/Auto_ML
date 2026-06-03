#!/bin/bash
# Проверка здоровья всех сервисов AutoML

echo "Проверка здоровья сервисов AutoML..."
echo ""

cd "$(dirname "$0")/.."

# Определение команды docker compose
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif docker-compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo "Docker Compose не найден."
    exit 1
fi

# Статус контейнеров
echo "Статус контейнеров:"
$COMPOSE_CMD ps

echo ""
echo "Доступные сервисы:"
echo "AutoML UI:      http://localhost:8501"
echo "Documentation:  http://localhost:8000"
echo "MLFlow UI:      http://localhost:5050"
echo "PostgreSQL:    localhost:5432"
echo ""

echo "Проверка доступности сервисов:"

if curl -s -o /dev/null -w "%{http_code}" http://localhost:5050 | grep -q "200\|302"; then
    echo "MLFlow UI доступен"
else
    echo "MLFlow UI недоступен (возможно еще запускается)"
fi

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200\|302"; then
    echo "AutoML UI доступен"
else
    echo "AutoML UI недоступен (возможно еще запускается)"
fi

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "200\|302"; then
    echo "Documentation доступна"
else
    echo "Documentation недоступна"
fi

echo ""
echo "Полезные команды:"
echo "   - Логи MLFlow:     $COMPOSE_CMD logs -f mlflow-service"
echo "   - Логи AutoML:     $COMPOSE_CMD logs -f automl"
echo "   - Остановка:       $COMPOSE_CMD down"
echo "   - Полная очистка:  $COMPOSE_CMD down -v"
echo ""
