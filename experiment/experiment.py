import os
from XtrRT.data import Data
from moviepy.editor import VideoFileClip
from datetime import datetime
import time


def play_videos(directory, data):
    # Get all files in the directory
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]

    # Sort files based on their names or any other criteria (assuming numeric ordering)
    video_files.sort()

    # Loop through each video file
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        annotation = video_file.split('.')[0]
        data.add_annotation(annotation)

        print(f"Now playing: {annotation}")

        # Load the video file
        clip = VideoFileClip(video_path)

        # Play the video
        clip.preview()

        # Optionally, you can add a delay between videos
        time.sleep(2)  # Delay in seconds (e.g., 2 seconds)


if __name__ == '__main__':

    host_name = "127.0.0.1"
    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as="test.edf")
    data.start()

    data.add_annotation("Start recording")

    directory = 'experiment videos'
    play_videos(directory, data)

    data.add_annotation("data.start_time: " + str(data.start_time))

    data.add_annotation("Stop recording")
    data.stop()

    print(data.annotations)
    print('process terminated')
