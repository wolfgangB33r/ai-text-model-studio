FROM python:3.7
LABEL maintainer="Wolfgang Beer @wolfgangB33r"

EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run src/app.py