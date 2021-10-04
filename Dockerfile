FROM python:3.8-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

COPY bin/pipeline.py \
     README.md \
     get_dependencies.py \
     config.py \
     /code/

ADD api /code/api

RUN python get_dependencies.py && ls -l /code 

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "80"]
