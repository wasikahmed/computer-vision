import sys
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11m-pose.pt")

print(model.names)  # Print the class names

# Define the pairs of keypoints to connect
skeleton = [
    (0, 1), (1, 3), (3, 5), (5, 6), (2, 4), (4, 6),
    (7, 9), (9, 11), (8, 10), (10, 12),
    (5, 6), (5, 7), (6, 8), (7, 8),
    (7, 13), (8, 14),
    (13, 15), (15, 17), (14, 16), (16, 17)
]
# skeleton = [
#     (0, 1), (1, 3), (3, 5), (5, 6), (4, 6), (2, 4), (7, 8), (5, 7), (7, 8), (6, 8), (7, 9), (9, 11), (8, 10),
#     (7, 13), (13, 15), (15, 17), (16, 17), (14, 16), (8, 14)
# ]
# skeleton = [
#     (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
#     (1, 11), (11, 12), (12, 13), (0, 14), (0, 15), (14, 16), (15, 17)
# ]

# Open a connection to the camera
s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])

cap = cv2.VideoCapture(s)
win_name = "YOLOv11 Pose Estimation"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform pose estimation
    results = model(frame)

    # Draw the results on the frame
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data[0]  # Access the keypoints data
            points = [None] * len(keypoints)

            for i, keypoint in enumerate(keypoints):
                x, y, confidence = keypoint
                if confidence > 0.5:  # Only draw keypoints with high confidence
                    points[i] = (int(x), int(y))
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Draw the skeleton
            for pair in skeleton:
                partA, partB = pair
                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (255, 0, 255), 2)  # Purple color

    # Display the resulting frame
    cv2.imshow(win_name, frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
