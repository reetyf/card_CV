FROM python:3

WORKDIR /anvil

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD [ "python", "./app.py" ]