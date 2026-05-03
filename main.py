"""
Точка входа для запуска AutoML из командной строки
Пример: python main.py --config configs/default_config.yaml --data data.csv
"""

import argparse
import yaml
import json
from pathlib import Path

from automl_core.pipeline.config import PipelineConfig
from automl_core.pipeline.orchestrator import AutoMLPipeline


def main():
    parser = argparse.ArgumentParser(description="AutoML Expert System")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Путь к конфигу")
    parser.add_argument("--data", type=str, required=True, help="Путь к датасету")
    parser.add_argument("--output", type=str, default="report.json", help="Путь к отчету")
    parser.add_argument("--save-model", action="store_true", help="Сохранить лучшую модель")

    args = parser.parse_args()

    # Загрузка конфига
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = PipelineConfig(**config_dict)

    # Запуск пайплайна
    print(f"🚀 Запуск AutoML с конфигом: {args.config}")
    print(f"📁 Датасет: {args.data}")

    pipeline = AutoMLPipeline(config)
    report = pipeline.run(args.data)

    # Сохранение отчета
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Отчет сохранен: {args.output}")

    # Сохранение модели
    if args.save_model:
        model_path = "best_model.joblib"
        pipeline.save_best_model(model_path)
        print(f"💾 Модель сохранена: {model_path}")

    # Вывод результатов
    print("\n🏆 РЕЗУЛЬТАТЫ")
    for result in report.get("results", []):
        if "error" not in result:
            print(f"\nМодель: {result['model']}")
            for metric, value in result.get("metrics", {}).items():
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    print("\n✅ AutoML завершен!")


if __name__ == "__main__":
    main()