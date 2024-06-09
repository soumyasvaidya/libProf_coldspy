# an import that will cause the -X importtime output
# to be saved to a file
from . import it

import signal
import traceback
import os
from io import BytesIO
import time

from . import cct

import json
PROFILER_FILEPATH = __file__
is_lambda_runtime = "LAMBDA_TASK_ROOT" in os.environ and "LOCAL" not in os.environ

stats_extension = f"-{time.time()}.pkl"
stats = {}

samples = []
mycct = cct.CCT()

def _sample(_, frame):
    global samples
    stack = traceback.extract_stack(frame)
    samples += [stack]

def parse_import_times():
    # this will get the import_times at this point of the program
    # handler() -> cct.dump_stats() -> parse_import_times()
    data = it.get_import_times()

    import importlib.util
    import re

    matches = re.findall(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|(\s+)\s*(.+)", data)

    parsed_data = []
    for match in matches:
        file = None

        try:
            file = importlib.util.find_spec(match[3].strip()).origin
        except Exception:
            pass
        parsed_data += [
            # name, file, self, cumul, spaces
            (match[3].strip(), file, int(match[0]), int(match[1]), len(match[2]))
        ]

    return parsed_data

def dump_stats(app_id="unnamed"):
    import pickle
    global samples
    for stack in samples:
        for f in stack:
            if f.filename == PROFILER_FILEPATH:
                return
     
        mycct.add_sample(stack)
    
    app_id = app_id.replace("-", "_")
    filename = app_id + stats_extension

    if "import_times" not in stats:
        print("parsing import times")
        stats["import_times"] = parse_import_times()

    stats["cct"] = mycct.root.to_dict()

    print("stats_filename is:", filename)

    if is_lambda_runtime:

        # the following lines of code will be imported after the profiling
        # this helps in avoiding to measure the boto3 import of the profiler
        # case 1. they were used by the app
        # in this case they will show up in the above profiling
        # because they were imported during the runtime of the lambda function
        # and the cct.dump_stats() is at the end of the function
        # case 2. they were not used by the app
        # they will show up in the log stream after the profiling time is collected

        import boto3 
        s3 = boto3.client('s3')

        try:
            s3.upload_fileobj(BytesIO(pickle.dumps(stats)), "profiling-stats-exports", filename)
        except Exception as e:
            print(f"Error uploading binary data: {e}")
    else:
        print("saving stats")
        with open("./profiler-stats/" + filename, "wb") as f:
            f.write(pickle.dumps(stats))

# According to some looking up this is supposed to run when AWS tries to terminate 
# the function and then give it a 500ms duration to cleanup before forcefully shutting it down
def handle_sigterm(signum, frame):
    print("received SIGTERM signal")

# def start_profiling():
interval = 0.1
signal.signal(signal.SIGPROF, _sample)
signal.setitimer(signal.ITIMER_PROF, interval, interval)
signal.signal(signal.SIGTERM, handle_sigterm)

# atexit.register(lambda: signal.setitimer(signal.ITIMER_PROF, 0))
# atexit.register(dump_stats)