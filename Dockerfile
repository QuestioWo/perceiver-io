FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /app

RUN apt-get update
RUN apt-get install -y --no-install-recommends curl

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - --version 1.2.0b2

COPY poetry.lock .
COPY pyproject.toml .

RUN /opt/poetry/bin/poetry config virtualenvs.create false
RUN /opt/poetry/bin/poetry install --no-root --all-extras --without dev

COPY perceiver ./perceiver

ARG RUN_INFERENCE_PATH
COPY $RUN_INFERENCE_PATH .

ARG MODEL_PATH
COPY $MODEL_PATH /app/
RUN if [ /app/*.ckpt != /app/best.ckpt ]; then mv /app/*.ckpt /app/best.ckpt; fi

ARG COREGISTRATION_IMAGE_PATH
COPY $COREGISTRATION_IMAGE_PATH /app/
RUN if [ /app/*.nii.gz != /app/coregistration_image.nii.gz ]; then mv /app/*.nii.gz /app/coregistration_image.nii.gz; fi

CMD ["python", "run_inferences.py"]
