# sklearn-docs-rag

RAG-сервис: вопросы по документации scikit-learn с цитатами.

## Локальный запуск

    conda create -y -n sklearn-docs-rag python=3.11
    conda activate sklearn-docs-rag
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Документация: http://localhost:8000/docs
