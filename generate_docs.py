"""
Script to generate HTML documentation using pydoc.

Usage:
    python generate_docs.py
    
Output:
    docs/index.html - Main documentation page
"""

import pydoc
import os
from pathlib import Path


def generate_docs():
    """Generate HTML documentation for all modules."""
    
    # Создаём папку docs
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Список модулей для документации
    modules = [
        "automl_core",
        "automl_core.data.loader",
        "automl_core.data.analyzer",
        "automl_core.preprocessing.cleaner",
        "automl_core.preprocessing.encoder",
        "automl_core.models.registry",
        "automl_core.models.trainer",
        "automl_core.models.tuner",
        "automl_core.evaluation.metrics",
        "automl_core.pipeline.config",
        "automl_core.pipeline.orchestrator",
        "interface.app",
    ]
    
    print("📚 Генерация документации...")
    
    # Генерируем HTML для каждого модуля
    for module in modules:
        try:
            pydoc.writedoc(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ⚠️  {module} - {e}")
    
    # Перемещаем файлы в папку docs
    for file in Path(".").glob("*.html"):
        if file.name != "index.html":
            file.rename(docs_dir / file.name)
    
    # Создаём главную страницу index.html
    index_html = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Expert System - Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .module-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .module-card {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .module-card a {
            font-weight: bold;
            font-size: 1.1em;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AutoML Expert System</h1>
        <p><strong>Documentation</strong> | Version 1.0.0</p>
        
        <h2>📖 Описание проекта</h2>
        <p>
            Модульная система автоматического машинного обучения с поддержкой:
        </p>
        <ul>
            <li>Загрузки и валидации данных</li>
            <li>Предобработки (очистка, кодирование, масштабирование)</li>
            <li>Обучения моделей с оптимизацией гиперпараметров</li>
            <li>Оценки и сравнения результатов</li>
            <li>Streamlit интерфейса</li>
        </ul>
        
        <h2>📦 Модули</h2>
        <div class="module-list">
            <div class="module-card">
                <a href="automl_core.html">automl_core</a>
                <p>Основной пакет системы</p>
            </div>
            <div class="module-card">
                <a href="automl_core.data.loader.html">automl_core.data.loader</a>
                <p>Загрузка и валидация данных</p>
            </div>
            <div class="module-card">
                <a href="automl_core.data.analyzer.html">automl_core.data.analyzer</a>
                <p>EDA и анализ данных</p>
            </div>
            <div class="module-card">
                <a href="automl_core.preprocessing.cleaner.html">automl_core.preprocessing.cleaner</a>
                <p>Очистка данных (пропуски, выбросы)</p>
            </div>
            <div class="module-card">
                <a href="automl_core.preprocessing.encoder.html">automl_core.preprocessing.encoder</a>
                <p>Кодирование и масштабирование</p>
            </div>
            <div class="module-card">
                <a href="automl_core.models.registry.html">automl_core.models.registry</a>
                <p>Реестр доступных моделей</p>
            </div>
            <div class="module-card">
                <a href="automl_core.models.trainer.html">automl_core.models.trainer</a>
                <p>Обучение и оценка моделей</p>
            </div>
            <div class="module-card">
                <a href="automl_core.models.tuner.html">automl_core.models.tuner</a>
                <p>Оптимизация гиперпараметров (Optuna)</p>
            </div>
            <div class="module-card">
                <a href="automl_core.evaluation.metrics.html">automl_core.evaluation.metrics</a>
                <p>Метрики оценки (Accuracy, F1, RMSE, etc.)</p>
            </div>
            <div class="module-card">
                <a href="automl_core.pipeline.config.html">automl_core.pipeline.config</a>
                <p>Конфигурация пайплайна (Pydantic)</p>
            </div>
            <div class="module-card">
                <a href="automl_core.pipeline.orchestrator.html">automl_core.pipeline.orchestrator</a>
                <p>Оркестратор всего пайплайна</p>
            </div>
            <div class="module-card">
                <a href="interface.app.html">interface.app</a>
                <p>Streamlit интерфейс</p>
            </div>
        </div>
        
        <h2>🚀 Быстрый старт</h2>
        <pre><code># Установка зависимостей
pip install -r requirements.txt

# Запуск интерфейса
streamlit run interface/app.py

# Запуск из CLI
python main.py --data dataset.csv --config configs/default_config.yaml

# Генерация документации
python generate_docs.py</code></pre>
        
        <h2>📝 Пример использования</h2>
        <pre><code>from automl_core.pipeline import PipelineConfig, AutoMLPipeline

# Создание конфигурации
config = PipelineConfig(
    target_column="target",
    task_type="classification",
    models=[
        {"name": "catboost", "tune_hyperparams": True, "n_trials": 50}
    ]
)

# Запуск пайплайна
pipeline = AutoMLPipeline(config)
report = pipeline.run("data.csv")

# Результаты
print(report["best_model"])</code></pre>
        
        <div class="footer">
            <p><strong>AutoML Expert System v1.0.0</strong></p>
            <p>Готово для интеграции с AI-агентами 🤖</p>
            <p>Лицензия: MIT</p>
        </div>
    </div>
</body>
</html>"""
    
    with open(docs_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    print(f"\n Документация сгенерирована в папке: {docs_dir.absolute()}")
    print(f" Откройте: {docs_dir.absolute() / 'index.html'}")


if __name__ == "__main__":
    generate_docs()