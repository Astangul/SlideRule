FROM python:3.11-slim-bookworm

ADD . /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl


RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt


EXPOSE 8501


ENTRYPOINT ["streamlit", "run", "00_👋_SlideRule_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
