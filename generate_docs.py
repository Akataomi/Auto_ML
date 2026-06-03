#!/usr/bin/env python3
"""
Генератор документации для AutoML проекта.
Сбалансированный стиль с компактной подачей.
"""

import os
import sys
import inspect
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

CSS_STYLES = """
:root {
    --primary: #2c3e50;
    --secondary: #34495e;
    --accent: #3498db;
    --highlight: #e74c3c;
    --success: #27ae60;
    --warning: #f39c12;
    --text: #2c3e50;
    --text-muted: #7f8c8d;
    --border: #e1e8ed;
    --bg: #ecf0f1;
    --card-bg: #ffffff;
    --code-bg: #f8f9fa;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--text);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

.container { max-width: 1200px; margin: 0 auto; }

header {
    background: var(--card-bg);
    padding: 35px 30px;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    margin-bottom: 30px;
    text-align: center;
}

header h1 {
    font-size: 2.2em;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 10px;
}

header .subtitle {
    color: var(--text-muted);
    font-size: 1.1em;
}

nav {
    background: var(--card-bg);
    padding: 18px 25px;
    margin-bottom: 30px;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

nav a {
    color: var(--accent);
    text-decoration: none;
    margin-right: 22px;
    font-weight: 500;
    transition: color 0.3s;
}

nav a:hover { color: var(--highlight); }

.section {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.section h2 {
    color: var(--primary);
    font-size: 1.5em;
    font-weight: 600;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 3px solid var(--accent);
}

.section h3 {
    color: var(--secondary);
    font-size: 1.2em;
    margin: 25px 0 15px 0;
}

/* Module cards */
.module-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin: 20px 0;
}

.module-link {
    display: block;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
    text-decoration: none;
    color: var(--text);
    transition: all 0.3s;
}

.module-link:hover {
    border-color: var(--accent);
    box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    transform: translateX(3px);
}

.module-link h3 {
    color: var(--accent);
    font-size: 1.05em;
    margin-bottom: 6px;
    font-family: 'Courier New', monospace;
}

.module-link p {
    color: var(--text-muted);
    font-size: 0.9em;
    margin: 0;
}

/* Class definitions - компактные */
.class-def {
    background: #f8f9fa;
    border: 1px solid var(--border);
    border-left: 4px solid var(--highlight);
    border-radius: 8px;
    padding: 18px;
    margin: 15px 0;
}

.class-def h5 {
    color: var(--primary);
    font-family: 'Courier New', monospace;
    font-size: 1em;
    margin-bottom: 10px;
    font-weight: 600;
}

.class-def .docstring {
    color: var(--text-muted);
    margin-bottom: 12px;
    font-size: 0.95em;
}

/* Methods - компактные с ограничением высоты */
.method {
    background: #fafbfc;
    border: 1px solid var(--border);
    border-left: 3px solid var(--success);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
}

.method summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--success);
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
    padding: 5px 0;
}

.method summary:hover {
    color: var(--accent);
}

.method .signature {
    font-family: 'Courier New', monospace;
    background: #e8f4f8;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 8px 0;
    font-size: 0.85em;
    border: 1px solid #d1e3f0;
}

.method .docstring {
    color: var(--text-muted);
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 0.9em;
    line-height: 1.6;
    max-height: 150px;
    overflow-y: auto;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background: var(--card-bg);
    font-size: 0.9em;
    border-radius: 8px;
    overflow: hidden;
}

th, td {
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

th {
    background: var(--primary);
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8em;
}

tr:hover { background: #f8f9fa; }

code {
    font-family: 'Courier New', monospace;
    background: var(--code-bg);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
    color: #c7254e;
}

.back-button {
    display: inline-block;
    background: var(--accent);
    color: white;
    padding: 12px 24px;
    border-radius: 25px;
    text-decoration: none;
    font-weight: 600;
    margin-bottom: 20px;
    transition: all 0.3s;
}

.back-button:hover {
    background: var(--primary);
    transform: translateY(-2px);
}

footer {
    text-align: center;
    padding: 30px 20px;
    color: rgba(255,255,255,0.9);
    font-size: 0.9em;
}

/* Category cards */
.category-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.category-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-decoration: none;
    transition: all 0.3s;
}

.category-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
}

.category-card h3 {
    color: white;
    margin-bottom: 10px;
    font-size: 1.15em;
}

.category-card p {
    color: rgba(255,255,255,0.9);
    font-size: 0.85em;
    line-height: 1.5;
}

/* Model items */
.model-item {
    background: #f8f9fa;
    padding: 12px 15px;
    border-radius: 6px;
    border-left: 3px solid var(--accent);
    margin: 8px 0;
}

.model-item strong {
    color: var(--primary);
    display: block;
    font-family: 'Courier New', monospace;
    font-size: 0.95em;
}

.model-item small {
    color: var(--text-muted);
    font-size: 0.85em;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 0.8em;
    font-weight: 600;
    margin-left: 10px;
}

.badge-primary { background: #3498db; color: white; }
.badge-success { background: #27ae60; color: white; }
.badge-warning { background: #f39c12; color: white; }
.badge-accent { background: #e74c3c; color: white; }
"""

def get_module_docstring(module):
    return inspect.getdoc(module) or "Нет документации"

def get_classes(module):
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            classes.append({
                'name': name,
                'doc': inspect.getdoc(obj) or "Нет документации",
                'methods': get_methods(obj),
                'bases': [b.__name__ for b in obj.__bases__ if b.__name__ != 'object']
            })
    return classes

def get_methods(cls):
    """Получить только собственные методы класса (не унаследованные)"""
    methods = []
    
    # Методы которые точно не показываем (унаследованные от torch.nn.Module и object)
    EXCLUDED_METHODS = {
        '__call__', '__delattr__', '__dir__', '__eq__', '__format__', 
        '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__',
        '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__',
        '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',
        '__str__', '__subclasshook__', '__weakref__',
        # torch.nn.Module методы
        'add_module', 'apply', 'bfloat16', 'buffers', 'children', 'compile',
        'cpu', 'cuda', 'double', 'eval', 'extra_repr', 'float', 'get_buffer',
        'get_extra_state', 'get_parameter', 'get_submodule', 'half', 'ipu',
        'load_state_dict', 'modules', 'mtia', 'named_buffers', 'named_children',
        'named_modules', 'named_parameters', 'parameters', 'register_backward_hook',
        'register_buffer', 'register_forward_hook', 'register_forward_pre_hook',
        'register_full_backward_hook', 'register_full_backward_pre_hook',
        'register_load_state_dict_post_hook', 'register_load_state_dict_pre_hook',
        'register_module', 'register_parameter', 'register_state_dict_post_hook',
        'register_state_dict_pre_hook', 'requires_grad_', 'set_extra_state',
        'set_submodule', 'share_memory', 'state_dict', 'to', 'to_empty',
        'train', 'type', 'xpu', 'zero_grad'
    }
    
    for name, obj in inspect.getmembers(cls, inspect.isfunction):
        # Пропускаем унаследованные методы
        if name in EXCLUDED_METHODS:
            continue
        # Пропускаем методы начинающиеся с _ кроме __init__
        if name.startswith('_') and name != '__init__':
            continue
        # Проверяем что метод определен в этом классе, а не унаследован
        if name not in cls.__dict__:
            continue
            
        try:
            sig = inspect.signature(obj)
            methods.append({
                'name': name,
                'signature': str(sig),
                'doc': inspect.getdoc(obj) or "Нет документации"
            })
        except (ValueError, TypeError):
            methods.append({
                'name': name,
                'signature': '(...)',
                'doc': inspect.getdoc(obj) or "Нет документации"
            })
    return methods

def get_functions(module):
    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:
            try:
                sig = inspect.signature(obj)
                functions.append({
                    'name': name,
                    'signature': str(sig),
                    'doc': inspect.getdoc(obj) or "Нет документации"
                })
            except (ValueError, TypeError):
                functions.append({
                    'name': name,
                    'signature': '(...)',
                    'doc': inspect.getdoc(obj) or "Нет документации"
                })
    return functions

def get_constants(module):
    constants = []
    for name, obj in inspect.getmembers(module):
        if name.isupper() and not inspect.ismodule(obj) and not inspect.isfunction(obj) and not inspect.isclass(obj):
            constants.append({
                'name': name,
                'value': str(obj)[:100]
            })
    return constants

def generate_module_html(module_name, module, output_dir):
    classes = get_classes(module)
    functions = get_functions(module)
    constants = get_constants(module)
    docstring = get_module_docstring(module)
    
    parts = module_name.split('.')
    parent_path = '/'.join(['..'] * len(parts))
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{module_name} - AutoML Documentation</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <a href="{parent_path}/index.html" class="back-button">← На главную</a>
        
        <header>
            <h1>📦 {module_name}</h1>
            <p class="subtitle">{docstring}</p>
        </header>
        
        <nav>
            <a href="#classes">📚 Классы</a>
            <a href="#functions">⚡ Функции</a>
            <a href="#constants">🔧 Константы</a>
        </nav>
"""
    
    if classes:
        html += """
        <section class="section" id="classes">
            <h2>📚 Классы</h2>
"""
        for cls in classes:
            bases_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
            html += f"""
            <div class="class-def">
                <h5>class {cls['name']}{bases_str}:</h5>
                <p class="docstring">{cls['doc'][:200]}{'...' if len(cls['doc']) > 200 else ''}</p>
"""
            if cls['methods']:
                for method in cls['methods']:
                    html += f"""
                <details class="method">
                    <summary>{method['name']}{method['signature']}</summary>
                    <div class="signature">{method['name']}{method['signature']}</div>
                    <p class="docstring">{method['doc'][:300]}{'...' if len(method['doc']) > 300 else ''}</p>
                </details>
"""
            html += """
            </div>
"""
        html += """
        </section>
"""
    
    if functions:
        html += """
        <section class="section" id="functions">
            <h2>⚡ Функции</h2>
"""
        for func in functions:
            html += f"""
            <details class="method">
                <summary>{func['name']}{func['signature']}</summary>
                <div class="signature">{func['name']}{func['signature']}</div>
                <p class="docstring">{func['doc'][:300]}{'...' if len(func['doc']) > 300 else ''}</p>
            </details>
"""
        html += """
        </section>
"""
    
    if constants:
        html += """
        <section class="section" id="constants">
            <h2>🔧 Константы</h2>
            <table>
                <thead>
                    <tr>
                        <th>Имя</th>
                        <th>Значение</th>
                    </tr>
                </thead>
                <tbody>
"""
        for const in constants:
            html += f"""
                    <tr>
                        <td><code>{const['name']}</code></td>
                        <td><code>{const['value']}</code></td>
                    </tr>
"""
        html += """
                </tbody>
            </table>
        </section>
"""
    
    html += f"""
        <footer>
            <p>AutoML Framework Documentation | Сгенерировано {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    output_path = output_dir / f"{module_name.replace('.', '_')}.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ {module_name} → {output_path.name}")
    return output_path.name

def generate_user_guides_html(output_dir):
    """Генерация страниц с руководствами пользователя"""
    
    # Getting Started Guide
    getting_started = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Быстрый старт - AutoML Documentation</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-button">← На главную</a>
        
        <header>
            <h1>🚀 Быстрый старт</h1>
            <p class="subtitle">Запуск AutoML проекта за 5 минут</p>
        </header>
        
        <nav>
            <a href="#requirements">Требования</a>
            <a href="#installation">Установка</a>
            <a href="#docker">Docker запуск</a>
            <a href="#local">Локальный запуск</a>
            <a href="#troubleshooting">Troubleshooting</a>
        </nav>
        
        <section class="section" id="requirements">
            <h2>📋 Требования</h2>
            <ul>
                <li><strong>Python:</strong> 3.10+</li>
                <li><strong>Docker Desktop:</strong> для контейнеризации</li>
                <li><strong>RAM:</strong> минимум 8GB (рекомендуется 16GB)</li>
                <li><strong>OS:</strong> macOS, Linux, Windows (WSL2)</li>
            </ul>
        </section>
        
        <section class="section" id="installation">
            <h2>📦 Установка</h2>
            <h3>1. Клонирование репозитория</h3>
            <pre><code>git clone &lt;repository-url&gt;
cd Auto_ML</code></pre>
            
            <h3>2. Создание виртуального окружения</h3>
            <pre><code>python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\\Scripts\\activate   # Windows</code></pre>
            
            <h3>3. Установка зависимостей</h3>
            <pre><code>pip install -r requirements.txt
pip install -e .</code></pre>
        </section>
        
        <section class="section" id="docker">
            <h2>🐳 Запуск через Docker (Рекомендуется)</h2>
            
            <h3>1. Запуск всех сервисов</h3>
            <pre><code>docker compose up -d --build</code></pre>
            
            <h3>2. Проверка статуса</h3>
            <pre><code>docker compose ps</code></pre>
            
            <h3>3. Доступ к сервисам</h3>
            <table>
                <thead>
                    <tr>
                        <th>Сервис</th>
                        <th>URL</th>
                        <th>Описание</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>AutoML UI</strong></td>
                        <td><a href="http://localhost:8501">http://localhost:8501</a></td>
                        <td>Веб-интерфейс для обучения моделей</td>
                    </tr>
                    <tr>
                        <td><strong>MLFlow UI</strong></td>
                        <td><a href="http://localhost:5050">http://localhost:5050</a></td>
                        <td>Мониторинг экспериментов</td>
                    </tr>
                    <tr>
                        <td><strong>Документация</strong></td>
                        <td><a href="http://localhost:8000">http://localhost:8000</a></td>
                        <td>API документация</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>4. Остановка сервисов</h3>
            <pre><code>docker compose down</code></pre>
        </section>
        
        <section class="section" id="local">
            <h2>💻 Локальный запуск (без Docker)</h2>
            
            <h3>1. Запуск AutoML UI</h3>
            <pre><code>streamlit run interface/app.py</code></pre>
            
            <h3>2. Запуск MLFlow (опционально)</h3>
            <pre><code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code></pre>
        </section>
        
        <section class="section" id="troubleshooting">
            <h2>🔧 Troubleshooting</h2>
            
            <h3>Ошибка: "Address already in use"</h3>
            <p>Освободите порты 8501, 5050, 8000 или измените их в docker-compose.yml</p>
            
            <h3>MLFlow не запускается</h3>
            <pre><code>docker compose logs mlflow-service</code></pre>
            
            <h3>Просмотр логов</h3>
            <pre><code>docker compose logs -f automl
docker compose logs -f mlflow-service</code></pre>
        </section>
        
        <footer>
            <p>AutoML Framework Documentation | Быстрый старт</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # MLFlow Integration Guide
    mlflow_guide = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLFlow Integration - AutoML Documentation</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-button">← На главную</a>
        
        <header>
            <h1>📊 MLFlow Integration</h1>
            <p class="subtitle">Полное руководство по логированию экспериментов</p>
        </header>
        
        <nav>
            <a href="#overview">Обзор</a>
            <a href="#architecture">Архитектура</a>
            <a href="#what-logged">Что логируется</a>
            <a href="#viewing">Просмотр</a>
            <a href="#examples">Примеры</a>
        </nav>
        
        <section class="section" id="overview">
            <h2>📋 Обзор</h2>
            <p>MLFlow автоматически логирует все эксперименты AutoML:</p>
            <ul>
                <li>✅ Параметры предобработки данных</li>
                <li>✅ Метрики всех моделей</li>
                <li>✅ Сами модели (в нативном формате)</li>
                <li>✅ Важность признаков</li>
                <li>✅ Графики обучения</li>
                <li>✅ Матрицы ошибок (для классификации)</li>
            </ul>
        </section>
        
        <section class="section" id="architecture">
            <h2>🏗️ Архитектура хранения</h2>
            
            <h3>PostgreSQL → Метаданные</h3>
            <ul>
                <li>Названия экспериментов и запусков</li>
                <li>Параметры (params)</li>
                <li>Метрики (metrics)</li>
                <li>Теги и временные метки</li>
            </ul>
            
            <h3>Docker Volume → Артефакты</h3>
            <ul>
                <li>Модели (catboost, xgboost, lightgbm, pytorch)</li>
                <li>Графики обучения (PNG)</li>
                <li>Матрицы ошибок (PNG)</li>
                <li>CSV с важностью признаков</li>
            </ul>
            
            <h3>Структура артефактов</h3>
            <pre><code>/mlflow/artifacts/
├── models/
│   ├── catboost/
│   ├── xgboost/
│   ├── lightgbm/
│   └── deep_mlp/
├── feature_importance/
│   ├── catboost/feature_importance.csv
│   ├── xgboost/feature_importance.csv
│   └── lightgbm/feature_importance.csv
├── training_curves/
│   └── deep_mlp.png
└── confusion_matrices/
    └── catboost.png</code></pre>
        </section>
        
        <section class="section" id="what-logged">
            <h2>📊 Что логируется в MLFlow</h2>
            
            <h3>1. Параметры эксперимента</h3>
            <table>
                <thead>
                    <tr>
                        <th>Параметр</th>
                        <th>Описание</th>
                        <th>Пример</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code>task_type</code></td>
                        <td>Тип задачи</td>
                        <td>regression, classification</td>
                    </tr>
                    <tr>
                        <td><code>metric</code></td>
                        <td>Целевая метрика</td>
                        <td>accuracy, rmse</td>
                    </tr>
                    <tr>
                        <td><code>target_column</code></td>
                        <td>Имя целевой колонки</td>
                        <td>target, price</td>
                    </tr>
                    <tr>
                        <td><code>num_models</code></td>
                        <td>Количество моделей</td>
                        <td>4</td>
                    </tr>
                    <tr>
                        <td><code>fill_strategy</code></td>
                        <td>Стратегия заполнения пропусков</td>
                        <td>median, mean</td>
                    </tr>
                    <tr>
                        <td><code>encode_strategy</code></td>
                        <td>Стратегия кодирования</td>
                        <td>onehot, label</td>
                    </tr>
                    <tr>
                        <td><code>scale</code></td>
                        <td>Масштабирование</td>
                        <td>True, False</td>
                    </tr>
                </tbody>
            </table>
            
            <h3>2. Метрики моделей (с префиксами)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Тип задачи</th>
                        <th>Метрики</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Регрессия</strong></td>
                        <td>
                            <code>catboost_rmse</code>, <code>xgboost_rmse</code>, <code>lightgbm_rmse</code>, <code>deep_mlp_rmse</code><br>
                            <code>catboost_mse</code>, <code>xgboost_mse</code>, <code>lightgbm_mse</code>, <code>deep_mlp_mse</code><br>
                            <code>catboost_mae</code>, <code>xgboost_mae</code>, <code>lightgbm_mae</code>, <code>deep_mlp_mae</code><br>
                            <code>catboost_r2</code>, <code>xgboost_r2</code>, <code>lightgbm_r2</code>, <code>deep_mlp_r2</code>
                        </td>
                    </tr>
                    <tr>
                        <td><strong>Классификация</strong></td>
                        <td>
                            <code>catboost_accuracy</code>, <code>xgboost_accuracy</code>, <code>lightgbm_accuracy</code>, <code>deep_mlp_accuracy</code><br>
                            <code>catboost_precision</code>, <code>xgboost_precision</code>, <code>lightgbm_precision</code>, <code>deep_mlp_precision</code><br>
                            <code>catboost_recall</code>, <code>xgboost_recall</code>, <code>lightgbm_recall</code>, <code>deep_mlp_recall</code><br>
                            <code>catboost_f1</code>, <code>xgboost_f1</code>, <code>lightgbm_f1</code>, <code>deep_mlp_f1</code>
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <h3>3. Важность признаков</h3>
            <p>Для каждой модели сохраняются:</p>
            <ul>
                <li><strong>Метрики:</strong> <code>catboost_importance_bmi</code>, <code>xgboost_importance_s5</code>, etc. (топ-20)</li>
                <li><strong>Артефакт:</strong> <code>feature_importance/catboost/feature_importance.csv</code>, <code>feature_importance/xgboost/feature_importance.csv</code></li>
            </ul>
            
            <h3>4. Графики обучения</h3>
            <p>Для итеративных моделей (Deep MLP, CatBoost, XGBoost, LightGBM):</p>
            <ul>
                <li><strong>Артефакт:</strong> <code>training_curves/deep_mlp.png</code>, <code>training_curves/catboost.png</code></li>
                <li>Два графика: loss curves + metric curves</li>
            </ul>
            
            <h3>5. Матрицы ошибок</h3>
            <p>Только для классификации:</p>
            <ul>
                <li><strong>Артефакт:</strong> <code>confusion_matrices/catboost.png</code>, <code>confusion_matrices/xgboost.png</code></li>
            </ul>
        </section>
        
        <section class="section" id="viewing">
            <h2>👁️ Просмотр результатов</h2>
            
            <h3>Через MLFlow UI</h3>
            <ol>
                <li>Откройте <a href="http://localhost:5050">http://localhost:5050</a></li>
                <li>Кликните на эксперимент (например, <code>AutoML_regression</code>)</li>
                <li>Кликните на запуск (run)</li>
                <li>Вкладка <strong>Parameters</strong> — параметры предобработки</li>
                <li>Вкладка <strong>Metrics</strong> — метрики всех моделей</li>
                <li>Вкладка <strong>Artifacts</strong> — модели, графики, CSV</li>
            </ol>
            
            <h3>Сравнение запусков</h3>
            <ol>
                <li>В списке экспериментов выберите несколько запусков</li>
                <li>Нажмите <strong>Compare</strong></li>
                <li>Сравните метрики и параметры</li>
            </ol>
        </section>
        
        <section class="section" id="examples">
            <h2>💡 Примеры использования</h2>
            
            <h3>Пример 1: Базовый запуск</h3>
            <pre><code># В AutoML UI:
1. Выберите датасет (Diabetes)
2. Тип задачи: Regression
3. Модели: CatBoost, XGBoost, LightGBM, Deep MLP
4. Нажмите "Запустить обучение"

# В MLFlow UI:
- Откройте http://localhost:5050
- Найдите эксперимент AutoML_regression
- Кликните на последний запуск
- Проверьте метрики и артефакты</code></pre>
            
            <h3>Пример 2: Сравнение экспериментов</h3>
            <pre><code># Запустите 2 эксперимента с разными параметрами:
Эксперимент 1: scale=True, encode_strategy=onehot
Эксперимент 2: scale=False, encode_strategy=label

# В MLFlow UI:
- Выберите оба запуска
- Нажмите Compare
- Сравните r2, rmse, mae метрики</code></pre>
            
            <h3>Пример 3: Загрузка модели</h3>
            <pre><code>import mlflow

# Загрузка модели из MLFlow
model_uri = "runs:/&lt;run_id&gt;/catboost"
model = mlflow.pyfunc.load_model(model_uri)

# Предсказание
predictions = model.predict(test_data)</code></pre>
        </section>
        
        <footer>
            <p>AutoML Framework Documentation | MLFlow Integration</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Сохранение файлов
    with open(output_dir / "getting_started.html", 'w', encoding='utf-8') as f:
        f.write(getting_started)
    print("✅ getting_started.html")
    
    with open(output_dir / "mlflow_integration.html", 'w', encoding='utf-8') as f:
        f.write(mlflow_guide)
    print("✅ mlflow_integration.html")


def generate_index_html(modules_by_category, output_dir):
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Framework - Documentation</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AutoML Framework</h1>
            <p class="subtitle">Полная документация проекта</p>
            <p style="margin-top: 15px; color: #95a5a6;">Сгенерировано: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>
        
        <nav>
            <a href="getting_started.html">🚀 Быстрый старт</a>
            <a href="mlflow_integration.html">📊 MLFlow</a>
            <a href="#core">🎯 Ядро</a>
            <a href="#models">🧠 Модели</a>
            <a href="#training">📈 Обучение</a>
            <a href="#preprocessing">🔧 Предобработка</a>
            <a href="#utils">🛠️ Утилиты</a>
            <a href="#interface">🖥️ Интерфейс</a>
        </nav>
"""
    
    category_descriptions = {
        'core': {'title': '🎯 Ядро (Core)', 'desc': 'Конфигурация, константы, пайплайны', 'badge': 'badge-primary'},
        'models': {'title': '🧠 Модели (Models)', 'desc': 'Реестр моделей, ML/DL реализации', 'badge': 'badge-success'},
        'training': {'title': '📈 Обучение (Training)', 'desc': 'Тренер, тюнинг, evaluation', 'badge': 'badge-warning'},
        'preprocessing': {'title': '🔧 Предобработка', 'desc': 'Очистка, кодирование данных', 'badge': 'badge-accent'},
        'data': {'title': '📊 Данные (Data)', 'desc': 'Загрузка, анализ, датасеты', 'badge': 'badge-primary'},
        'utils': {'title': '🛠️ Утилиты', 'desc': 'Вспомогательные функции', 'badge': 'badge-success'},
        'interface': {'title': '🖥️ Интерфейс', 'desc': 'Streamlit компоненты', 'badge': 'badge-warning'}
    }
    
    for category, modules in modules_by_category.items():
        if not modules:
            continue
        
        cat_info = category_descriptions.get(category, {'title': category, 'desc': '', 'badge': 'badge-primary'})
        
        html += f"""
        <section class="section" id="{category}">
            <h2>{cat_info['title']} <span class="badge {cat_info['badge']}">{len(modules)} модулей</span></h2>
            <p style="color: #7f8c8d; margin-bottom: 20px;">{cat_info['desc']}</p>
            
            <div class="category-grid">
"""
        
        for module_name, module in modules.items():
            docstring = get_module_docstring(module)
            docstring_preview = docstring[:100] + "..." if len(docstring) > 100 else docstring
            filename = f"{module_name.replace('.', '_')}.html"
            
            html += f"""
                <a href="{filename}" class="category-card">
                    <h3>📦 {module_name.split('.')[-1]}</h3>
                    <p>{docstring_preview}</p>
                </a>
"""
        
        html += """
            </div>
        </section>
"""
    
    try:
        from automl_core.models.registry import ModelRegistry
        registry = ModelRegistry()
        html += """
        <section class="section" id="models-showcase">
            <h2>🧠 Доступные модели</h2>
"""
        
        for task_type in ['classification', 'regression', 'clustering', 'anomaly_detection']:
            models = registry.get_available_models(task_type)
            if models:
                html += f"""
            <h3>{task_type.title()}</h3>
            <div class="module-list">
"""
                for model_name in models:
                    html += f"""
                <div class="model-item">
                    <strong>{model_name}</strong>
                    <small>Доступна для {task_type}</small>
                </div>
"""
                html += """
            </div>
"""
        
        html += """
        </section>
"""
    except Exception as e:
        print(f"⚠️ Не удалось получить список моделей: {e}")
    
    html += """
        <footer>
            <p>AutoML Framework Documentation | Built with ❤️ for Data Scientists</p>
        </footer>
    </div>
</body>
</html>
"""
    
    output_path = output_dir / "index.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ index.html → {output_path}")

def main():
    print("🚀 Генерация документации AutoML Framework...")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / "docs"
    output_dir.mkdir(exist_ok=True)
    
    for f in output_dir.glob("*.html"):
        f.unlink()
    print("🗑️  Очищена папка docs/")
    
    modules_by_category = {
        'core': {}, 'models': {}, 'training': {},
        'preprocessing': {}, 'data': {}, 'utils': {}, 'interface': {}
    }
    
    modules_to_doc = [
        'automl_core.config.constants',
        'automl_core.pipeline.config',
        'automl_core.pipeline.orchestrator',
        'automl_core.models.registry',
        'automl_core.models.base',
        'automl_core.models.mlp',
        'automl_core.models.deep_learning',
        'automl_core.training.trainer',
        'automl_core.tuning.optuna_tuner',
        'automl_core.evaluation.metrics',
        'automl_core.preprocessing.cleaner',
        'automl_core.preprocessing.encoder',
        'automl_core.data.loader',
        'automl_core.data.analyzer',
        'automl_core.data.datasets',
        'automl_core.utils.helpers',
        'interface.app',
        'interface.utils',
    ]
    
    for module_path in modules_to_doc:
        try:
            module = __import__(module_path, fromlist=[''])
            
            if 'config' in module_path or 'pipeline' in module_path:
                category = 'core'
            elif 'models' in module_path:
                category = 'models'
            elif 'training' in module_path or 'tuning' in module_path or 'evaluation' in module_path:
                category = 'training'
            elif 'preprocessing' in module_path:
                category = 'preprocessing'
            elif 'data' in module_path:
                category = 'data'
            elif 'utils' in module_path:
                category = 'utils'
            elif 'interface' in module_path:
                category = 'interface'
            else:
                category = 'core'
            
            modules_by_category[category][module_path] = module
            print(f"📦 Импортирован: {module_path}")
        except ImportError as e:
            print(f"⚠️  Пропущен {module_path}: {e}")
        except Exception as e:
            print(f"❌ Ошибка {module_path}: {e}")
    
    print("\n📄 Генерация страниц модулей...")
    print("=" * 60)
    
    for category, modules in modules_by_category.items():
        if modules:
            print(f"\n📁 Категория: {category}")
            for module_name, module in modules.items():
                try:
                    generate_module_html(module_name, module, output_dir)
                except Exception as e:
                    print(f"❌ Ошибка генерации {module_name}: {e}")
    
    print("\n🏠 Генерация главной страницы...")
    print("=" * 60)
    generate_index_html(modules_by_category, output_dir)
    
    print("\n📚 Генерация руководств пользователя...")
    print("=" * 60)
    generate_user_guides_html(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Документация сгенерирована!")
    print(f"📁 Папка: {output_dir.absolute()}")
    print("🌐 Открой: docs/index.html")
    print("\n📖 Руководства:")
    print("   - docs/getting_started.html (Быстрый старт)")
    print("   - docs/mlflow_integration.html (MLFlow)")
    print("=" * 60)

if __name__ == "__main__":
    main()
