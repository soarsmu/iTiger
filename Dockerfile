FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yaml .
COPY src/ .

RUN conda env create -f conda.yaml
RUN conda init bash
RUN echo "source activate issue-title-generator" > ~/.bashrc

WORKDIR /app/src

EXPOSE 8000
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "issue-title-generator", "uvicorn", "api:app", "--host", "0.0.0.0"]
