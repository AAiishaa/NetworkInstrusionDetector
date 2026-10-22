FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/best_model.joblib ${LAMBDA_TASK_ROOT}/models/best_model.joblib

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src

CMD ["intrusion_detector.app.handler"]