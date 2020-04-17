TEMPLATE = """
FROM python:3.7.7-slim

WORKDIR /app/

COPY ./requirements.txt /app

RUN pip install numpy
RUN pip install -r /app/requirements.txt

COPY ./ /app/

CMD ["python", "{name}_runner.py"]
"""


def get_dockerfile_template(name: str) -> str:
    return TEMPLATE.format(name=name)