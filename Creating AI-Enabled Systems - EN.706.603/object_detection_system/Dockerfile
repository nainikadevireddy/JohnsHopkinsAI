FROM python:3.11.7

RUN apt update && apt install -y ffmpeg 

RUN pip install ffmpeg-python opencv-python-headless

WORKDIR /app
COPY . /app

EXPOSE 23000

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]
