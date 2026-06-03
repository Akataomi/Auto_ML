#!/bin/bash
# Остановка всей инфраструктуры AutoML

set -e

echo "Остановка AutoML инфраструктуры..."
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

$COMPOSE_CMD down

echo "Инфраструктура остановлена"