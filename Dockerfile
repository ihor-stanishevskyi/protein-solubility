# syntax=docker/dockerfile:1

# Three stages. The order matters: whichever comes LAST is what `docker build .`
# produces when no target is named. Yesterday's sibling project put the Lambda
# stage at the end by accident and CI spent a run smoke-testing an image that
# can never answer an HTTP request. So `runtime` stays last, and CI names its
# target explicitly anyway.

# Which model gets baked in. Defaults to the stand-in so the image can be built
# and smoke-tested before the trained weights exist; CI passes the real one.
ARG MODEL_SOURCE=models/stand-in.onnx
# The card states what these weights are and what they scored. It travels with
# them so the service can never quote a number the loaded model did not earn.
ARG MODEL_CARD=models/stand-in.card.json

# ---------------------------------------------------------------- lambda ----
FROM public.ecr.aws/lambda/python:3.13 AS lambda
ARG MODEL_SOURCE
ARG MODEL_CARD
# Both files: requirements-lambda.txt starts with "-r requirements.txt", so
# copying only the first leaves pip looking for a file that is not there.
COPY requirements.txt requirements-lambda.txt ./
RUN pip install --no-cache-dir -r requirements-lambda.txt
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY ${MODEL_SOURCE} ${LAMBDA_TASK_ROOT}/app/model.onnx
COPY ${MODEL_CARD} ${LAMBDA_TASK_ROOT}/app/model.card.json
CMD ["app.lambda_handler.handler"]

# --------------------------------------------------------------- runtime ----
# Last on purpose: a plain `docker build .` should give the ordinary service.
FROM python:3.13-slim AS runtime
ARG MODEL_SOURCE
ARG MODEL_CARD
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY ${MODEL_SOURCE} ./app/model.onnx
COPY ${MODEL_CARD} ./app/model.card.json
# Nothing here needs root.
RUN useradd --system --uid 10001 service && chown -R service /srv
USER service
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4).status==200 else 1)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
