# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY ./webapp .

WORKDIR /app/webapp

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "leaderboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
