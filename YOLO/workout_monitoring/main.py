import cv2
from ultralytics import solutions

# Capture video from the default camera
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening camera"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init AIGym
gym = solutions.AIGym(
    show=True,  # Display the frame
    kpts=[6, 8, 10],  # keypoints index of person for monitoring specific exercise, by default it's for pushup
    model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
    # line_width=2,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = gym.monitor(im0)
    video_writer.write(im0)

cv2.destroyAllWindows()
video_writer.release()