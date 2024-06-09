# Replication Package of ColdSpy

## A Python Profiler to Guide Cold Start Optimization in Serverless Applications

## Abstract
Serverless computing abstracts away server management, enabling automatic scaling and efficient resource utilization. However, coldstart latency remains a significant challenge, affecting end-to-end performance. Our preliminary study reveals that inefficient library initialization and usage are major contributors to this latency in Python-based serverless applications. We introduce ColdSpy, a Python profiler that uses dynamic program analysis to identify inefficient library initializations. ColdSpy collects library usage data through statistical sampling and call-path profiling, then generates a report to guide developers in addressing four types of inefficiency patterns. Systematic evaluations on 15 serverless applications demonstrate that ColdSpy effectively identifies inefficiencies, achieving up to 2.26× speedup in cold-start execution time, and 1.51× reduction in memory usage.

## Replication

In the following sections, we describe how to use this replication package

### File Structure
- `profiler`: contains the ColdSpy profiler 
- `visualizer_guide`: contains step by step guidance explaining "How to use ColdSpy visualizer". The live visualizer url is given in the end of this file.
- `data`: contains the data-files
- `cdf_plots`: contains the cumulative distribution plots of both initialization and execution latency
  Each subdirectory is named after the application and each of them contains 2 files `initialization_latency.png` and `execution_latency.png`
- `applications`:
    - This directory contains all the applications we evaluate in this work
    - Each sub directory represents application names which include both original and optimized code
        - `original`: represents the original code of each applications
        - `optimized`: represents the optimized code of each applications
            - `hanlder.py`: Is the serverless function handler which is the main program file. It may also have names as `lambda_function.py`
            - `requirements.txt`: The dependencies for each application
            - `Dockerfile`: The command file that allow you build Docker images for the application
            - Some applications may contain specialized files for machine learning model, input dataset etc.

### Prerequisites
- Access to AWS Console
- Access to AWS services including Lambda, S3, CloudWatch, ECR with execution role
- Python 3.9 runtime
- AWS CLI
- Docker

### Dependency Installations
```
pip install -r ./requirements.txt --platform manylinux2014_x86_64 --target=./$(PACKAGE_DIRNAME) --implementation cp --python-version $(PYTHON_VERSION) --only-binary=:all: --upgrade
```

### Create Deployment Package
```
rm -rf $(ZIP_FILE_NAME)
cd $(PACKAGE_DIRNAME); zip -r ../$(ZIP_FILE_NAME) .; cd ..
zip -r $(ZIP_FILE_NAME) handler.py
```

### Create AWS Lambda Function
Use the command below to create lambda function but provide appropriate data to all the place holders.
```
aws lambda create-function --function-name $(LAMBDA_NAME) \
--runtime $(RUNTIME) --role $(ROLE_ARN) --handler $(HANDLER) \
--zip-file fileb://$(ZIP_FILE_NAME) --timeout $(TIMEOUT) --memory-size $(MEMORY_SIZE) \
--environment $(ENV_VARS) --region $(AWS_REGION)
```

### Run The Application
Following make command can be used to request lambda function. You must provide a valid `API_URL`
```
invoke-par:
	@echo "Invoking in parallel..."
	@for i in {1..$(INVOKE_COUNT)}; do \
		curl --request POST \
			--url $(API_URL) \
			--header 'Content-Type: application/json' & \
	done; wait
```

### ColdSpy Visualizer with Live Profile Data
All the profiling data for the applications that which ColdSpy optimizes are available here:
https://coldspy.netlify.app/