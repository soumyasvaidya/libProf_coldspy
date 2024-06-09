import os
import sys
import stat
import subprocess
# import imageio_ffmpeg as iioffmpeg
import ffmpeg
import shutil

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

image_name = "watermark.png"
video_name = "hi_chitanda_eru.mp4"
local_path = "/tmp"

# https://github.com/kkroening/ffmpeg-python
def to_video(duration):
    output = local_path + '/processed_hi_chitanda_eru.mp4'
    # Use ffmpeg-python to trim the first 10 seconds of the video
    try:
        video_input = ffmpeg.input(video_name)
        image_input  = ffmpeg.input(image_name)
        (
            ffmpeg
            .concat(
                video_input.trim(start_frame=0, end_frame=50),
                video_input.trim(start_frame=100, end_frame=150),
            )
            .overlay(image_input.hflip())
            .drawbox(50, 50, 120, 120, color='red', thickness=5)
            .output(output)
            .run(overwrite_output=True)
        )
    except Exception as e:
        return {
            "result": f"Error processing video: {str(e)}"
        }

    return "Video {} finished!".format(output)

def handler(event, context=None):
    duration = 5

    # Get the ffmpeg binary location from imageio_ffmpeg
    # ffmpeg_original_path = iioffmpeg.get_ffmpeg_exe()
    ffmpeg_original_path = "./imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2"
    # print(ffmpeg_original_path)

    # Copy the ffmpeg binary to /tmp and set executable permissions
    ffmpeg_tmp_path = local_path + "/ffmpeg"
    if not os.path.exists(ffmpeg_tmp_path):
        shutil.copy(ffmpeg_original_path, ffmpeg_tmp_path)
        os.chmod(ffmpeg_tmp_path, 0o755)  # Ensure executable permissions

    # Add the /tmp directory to the PATH
    os.environ["PATH"] = local_path + ":" + os.environ.get("PATH", "")

    # Process media
    result = to_video(duration)

    return {
        "result": result
    }

if __name__ == "__main__":
    event = {
        "duration": 10
    }

    print(handler(event))
