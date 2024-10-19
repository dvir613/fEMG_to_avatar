import os
from XtrRT.data import Data
from moviepy.editor import VideoFileClip
from datetime import datetime
import time
import pygame
from pygame.locals import QUIT


def play_videos(directory, data, trial_number):
    # Get all files in the directory
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]

    # Sort files based on their names or any other criteria (assuming numeric ordering)
    video_files.sort()

    # Loop through each video file
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        # TODO remove the following lines
        if '02_Introduction' in video_file:
            continue
        if '01_Info' in video_file:
            continue
        if '37_Credits' in video_file:
            continue
        if '36_Natural_smile' in video_file:
            continue
        if 'demonstration' in video_file:
            continue

        annotation = video_file.split('.')[0]
        data.add_annotation(annotation + f"_trial_{trial_number}")

        print(f"Now playing: {annotation}")

        # Load the video file
        clip = VideoFileClip(video_path)

        # Play the video
        clip.preview()



def free_behavior():
    # write on the screen: "free behavior"
    data.add_annotation("free_behavior")
    # Initialize Pygame
    pygame.init()
    # Set up the display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Experiment Display")
    # Set up the font
    font = pygame.font.Font(None, 64)
    # Render the text
    text = font.render("free behavior", True, (255, 255, 255))
    text_rect = text.get_rect(center=(400, 300))
    # Get the current time
    start_time = time.time()
    # Main loop
    while time.time() - start_time < 120:  # Run for 2 minutes (120 seconds)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((0, 0, 0))  # Fill the screen with black
        screen.blit(text, text_rect)
        pygame.display.flip()
    # Clean up Pygame
    pygame.quit()
    data.add_annotation("finished_free_behavior")



if __name__ == '__main__':

    host_name = "127.0.0.1"

    data_path = r"C:\Users\YH006_new\fEMG_to_avatar\data"
    participant_ID = "participant_03"
    session_number = 2
    participant_folder = os.path.join(data_path, participant_ID)
    session_folder = os.path.join(participant_folder, f"S{session_number}")
    if not os.path.exists(participant_folder):
        os.makedirs(participant_folder)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    edf_file_path = os.path.join(session_folder, f"{participant_ID}_S{session_number}.edf")

    port = 20001
    n_bytes = 1024
    data = Data(host_name, port, verbose=False, timeout_secs=15, save_as=edf_file_path)
    data.start()

    data.add_annotation("Start recording")
    directory = 'experiment videos'
    for trial_number in range(1, 4):
        play_videos(directory, data, str(trial_number))

    free_behavior()

    data.add_annotation("data_start_time: " + str(data.start_time))

    data.add_annotation("stop_recording")
    data.stop()

    print(data.annotations)
    print('process_terminated')
