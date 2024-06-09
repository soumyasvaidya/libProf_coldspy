# from profiler import dump_stats

from skimage import io
import urllib.request
# import skimage.segmentation as segmentation

def handler(event, context):
    urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/3/38/JPEG_example_JPG_RIP_001.jpg", "/tmp/hi.jpg")
    img = io.imread('/tmp/hi.jpg')
    print(img)

    # dump_stats("Skimage_numpy")

    return {"statusCode": 200, "body": "success"}