FROM python:3.11
WORKDIR /ImageHog
COPY ./requirements.txt /ImageHog/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /ImageHog/requirements.txt

COPY ./app /ImageHog/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV PYTHONPATH="${PYTHONPATH}:/ImageHog"

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

