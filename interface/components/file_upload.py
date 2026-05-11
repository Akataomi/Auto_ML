"""
File Upload Component

Handles file upload, preview, and basic statistics
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from interface.state import set_file, clear_file


def render_file_upload() -> bool:
    """
    Render file upload component.
    
    Returns:
        bool: True if file is loaded, False otherwise.
    """
    st.subheader("📁 Загрузка данных")
    
    uploaded_file = st.file_uploader(
        "Загрузите датасет (CSV)", 
        type=["csv"], 
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        if uploaded_file != st.session_state.get("last_uploaded_file"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                    tmp.flush()
                    os.fsync(tmp.fileno())

                try:
                    df = pd.read_csv(tmp_path)

                    if df.empty:
                        st.error("❌ Файл пустой или не содержит данных")
                        os.remove(tmp_path)
                        clear_file()
                        return False

                    set_file(uploaded_file, tmp_path, df)
                    
                except pd.errors.EmptyDataError:
                    st.error("❌ Файл пустой или некорректный CSV")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    clear_file()
                    return False
                except Exception as e:
                    st.error(f"❌ Ошибка чтения файла: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    clear_file()
                    return False
                    
            except Exception as e:
                st.error(f"❌ Ошибка сохранения файла: {str(e)}")
                clear_file()
                return False
        
        df = st.session_state.df_preview
        
        if df is None:
            st.error("❌ Ошибка загрузки данных")
            return False

        st.dataframe(df.head(), width="stretch")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Записей", df.shape[0])
        with col2:
            st.metric("Признаков", df.shape[1])
        with col3:
            st.metric("Пропуски", df.isna().sum().sum())
        
        return True
    else:
        st.info("👆 Загрузите CSV файл для начала работы")
        clear_file()
        return False