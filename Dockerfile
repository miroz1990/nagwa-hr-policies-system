FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefer-binary --default-timeout=1000 --retries=20 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/chat_ui.py"]