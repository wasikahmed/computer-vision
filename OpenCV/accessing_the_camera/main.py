import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = int(sys.argv[1])
    
source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while(cv2.waitKey(1) != 27): # 27 is the ASCII code for the ESC key
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(win_name, frame)
    
source.release()
cv2.destroyAllWindows()