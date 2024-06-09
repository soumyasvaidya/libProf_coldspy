FROM public.ecr.aws/lambda/python:3.9

COPY package ${LAMBDA_TASK_ROOT}/package
COPY handler.py ${LAMBDA_TASK_ROOT}/

# Set the Python path to include the copied packages
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/package

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handler" ]
