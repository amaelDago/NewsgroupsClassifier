FROM python:3.8-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

COPY bin/pipeline.py \
     README.md \
     run.sh \
     get_dependencies.py \
     config.py \
     /code/

ADD api /code/api
#ADD train_embedding /code/train_embedding
#ADD models /code/models

RUN python get_dependencies.py && ls -l /code
# RUN chmod +x /code/run.sh 

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "80"]
#CMD ["./run.sh"]