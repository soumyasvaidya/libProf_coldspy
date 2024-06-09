FROM public.ecr.aws/lambda/python:3.9

COPY .cache ${LAMBDA_TASK_ROOT}/.cache
COPY cloned_repositories ${LAMBDA_TASK_ROOT}/cloned_repositories
COPY package ${LAMBDA_TASK_ROOT}/package
COPY repo_archive_list.csv ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/

# Set the Python path to include the copied packages
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/package

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handler" ]
