# from profiler import dump_stats

import json
import urllib.parse
import boto3
import ocrmypdf
import uuid
import os

def apply_ocr_to_document_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    if "body" in event:
        event = json.loads(event["body"])

    region_name = event['awsRegion']
    s3 = boto3.client('s3', region_name=region_name)

    bucket = event['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['s3']['object']['key'], encoding='utf-8')
    pages = event['pages']
    uuidstr = str(uuid.uuid1())
    try:
        inputname = '/tmp/input' + uuidstr + '.pdf'
        outputname = '/tmp/output' + uuidstr + '.pdf'
        s3.download_file(Bucket=bucket, Key=key, Filename=inputname)
        ocrmypdf.ocr(inputname, outputname, pages=pages, force_ocr=True)
        s3.upload_file(inputname, bucket, key + '.bak')
        s3.upload_file(outputname, bucket, key)
        os.remove(inputname)
        os.remove(outputname)

    except Exception as e:
        print(e)
        raise e

    # dump_stats("lambda_OCRmyPDF")

    return {
        "statusCode": 200,
        "body": "success"
    }