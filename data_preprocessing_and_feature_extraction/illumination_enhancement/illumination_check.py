import cv2
import os


def check_brightness(video_path, threshold=100, duration=1):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_frames = frame_rate * duration
    consecutive_low_brightness_frames = 0

    for _ in range(target_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # illumination calculation（can be modified according to the requirement）
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray_frame)[0]

        # judge whether illumination is strong enough
        if brightness < threshold:
            consecutive_low_brightness_frames += 1
        else:
            consecutive_low_brightness_frames = 0

        # if there is one continuous second that has low brightness, it is classified to not bright enough
        if consecutive_low_brightness_frames >= frame_rate * duration:
            cap.release()

            return True

    cap.release()
    return False


def check_brightness_for_folder(folder_path, threshold=100, duration=1):
    videos_with_low_brightness = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(folder_path, filename)
            if check_brightness(video_path, threshold, duration):
                count += 1
                videos_with_low_brightness.append(filename)
    print(count)

    return videos_with_low_brightness


folder_path = ""
low_brightness_videos = check_brightness_for_folder(folder_path, threshold=100, duration=1)
output_file_path = "./low_brightness_videos.txt"
with open(output_file_path, "w") as output_file:
    for video_name in low_brightness_videos:
        print("Videos with low brightness:", video_name)
        output_file.write(video_name + "\n")

print("Videos with low brightness saved to:", output_file_path)
