import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path

from automl_core.pipeline.config import PipelineConfig, PreprocessingConfig, ModelConfig
from automl_core.pipeline.orchestrator import AutoMLPipeline
from automl_core.models.registry import ModelRegistry

# Конфиг страницы
st.set_page_config(page_title="AutoML Expert System", page_icon="🤖", layout="wide")

# Заголовок
st.title("🤖 AutoML Expert System")
st.markdown("---")

# Сайдбар с настройками
st.sidebar.header("⚙️ Настройки пайплайна")

uploaded_file = st.file_uploader("📁 Загрузите датасет (CSV)", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    df_preview = pd.read_csv(tmp_path)
    st.subheader("📊 Предпросмотр данных")
    st.dataframe(df_preview.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Записей", df_preview.shape[0])
    with col2:
        st.metric("Признаков", df_preview.shape[1])
    with col3:
        st.metric("Пропуски", df_preview.isna().sum().sum())

    target_col = st.selectbox("🎯 Целевая переменная", df_preview.columns.tolist())

    if df_preview[target_col].dtype in ["object", "category", "bool"]:
        task_type = "classification"
        st.info("📌 Определена задача: **Классификация**")
    else:
        task_type = "regression"
        st.info("📌 Определена задача: **Регрессия**")

    st.sidebar.subheader("Предобработка")
    fill_method = st.sidebar.selectbox("Заполнение пропусков", ["median", "mean", "most_frequent"])
    encode_method = st.sidebar.selectbox("Кодирование категориальных", ["onehot", "label"])
    scale = st.sidebar.checkbox("Масштабирование", value=True)
    handle_outliers = st.sidebar.checkbox("Обработка выбросов", value=False)

    st.sidebar.subheader("Модели")
    registry = ModelRegistry()
    available_models = registry.get_available_models(task_type)
    selected_models = st.sidebar.multiselect(
        "Выберите модели для обучения", available_models, default=["catboost", "lightgbm"][:len(available_models)]
    )

    st.sidebar.subheader("Оптимизация")
    tune_hyperparams = st.sidebar.checkbox("Оптимизация гиперпараметров (Optuna)", value=False)
    n_trials = st.sidebar.slider("Количество испытаний Optuna", 10, 100, 30) if tune_hyperparams else 30

    st.markdown("---")
    if st.button("🚀 Запустить AutoML", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Выберите хотя бы одну модель!")
        else:
            with st.spinner("⏳ Обучение моделей... Это может занять несколько минут"):
                try:
                    config = PipelineConfig(
                        target_column=target_col,
                        task_type=task_type,
                        preprocessing=PreprocessingConfig(
                            fill_missing=fill_method,
                            encode_categorical=encode_method,
                            scale=scale,
                            handle_outliers=handle_outliers,
                        ),
                        models=[
                            ModelConfig(name=m, tune_hyperparams=tune_hyperparams, n_trials=n_trials)
                            for m in selected_models
                        ],
                    )

                    pipeline = AutoMLPipeline(config)
                    report = pipeline.run(tmp_path)

                    st.success("✅ Обучение завершено!")

                    st.subheader("🏆 Результаты моделей")
                    if report.get("results"):
                        results_df = pd.DataFrame(
                            [
                                {
                                    "Модель": r.get("model"),
                                    "Accuracy": r.get("metrics", {}).get("accuracy", "-"),
                                    "F1": r.get("metrics", {}).get("f1", "-"),
                                    "AUC-ROC": r.get("metrics", {}).get("auc_roc", "-"),
                                    "RMSE": r.get("metrics", {}).get("rmse", "-"),
                                    "R²": r.get("metrics", {}).get("r2", "-"),
                                    "CV Mean": r.get("metrics", {}).get("cv_mean", "-"),
                                }
                                for r in report["results"]
                                if "error" not in r
                            ]
                        )
                        st.dataframe(results_df, use_container_width=True)

                    if pipeline.best_model and pipeline.best_model.feature_importance is not None:
                        st.subheader("📈 Важность признаков")
                        importance = pipeline.get_feature_importance()
                        if importance:
                            imp_df = pd.DataFrame(
                                {"Признак": list(importance.keys()), "Важность": list(importance.values())}
                            ).sort_values("Важность", ascending=False)
                            st.bar_chart(imp_df.set_index("Признак").head(15))

                    st.subheader("💾 Сохранение модели")
                    if st.button("Сохранить лучшую модель"):
                        save_path = f"best_model_{target_col}.joblib"
                        pipeline.save_best_model(save_path)
                        st.success(f"Модель сохранена: {save_path}")

                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
else:
    st.info("👆 Загрузите CSV файл для начала работы")

st.markdown("---")
st.markdown(
    """
    **AutoML Expert System v1.0** | 
    Поддерживаемые модели: CatBoost, LightGBM, XGBoost, Random Forest |
    Готово для интеграции с AI-агентами 🤖
    """
)