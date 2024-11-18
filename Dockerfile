# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/lykos98/adsai_pong.git

WORKDIR /app/adsai_pong

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "leaderboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
