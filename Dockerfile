FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./static /code/static

COPY ./main.py /code/

COPY ./test_main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4040", "--workers", "4"]
