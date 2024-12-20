import cv2
import sys
import os

from urllib.request import urlretrieve
from zipfile import ZipFile

def download_and_unzip(url, zip_file, folder):
    print(f'Downloading and extracting assets...', end='')

    if not os.path.exists(folder):
        os.makedirs(folder)
        
    try:
        # download the zip file if it doesn't exist
        if not os.path.exists(zip_file):
            urlretrieve(url, zip_file)
        
        # unzip the file
        with ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(folder)
        print('Done!')
        
    except Exception as e:
        print(f'Error: {e}')
        
# https://drive.google.com/file/d/1a1NuJl3jkOLR0NxvTVIUCBFO8AFdg-QF/view?usp=sharing
file_id = '1a1NuJl3jkOLR0NxvTVIUCBFO8AFdg-QF'
URL = r"https://drive.google.com/file/d/1a1NuJl3jkOLR0NxvTVIUCBFO8AFdg-QF/view?usp=sharing"

folder = os.path.join(os.getcwd(), 'assets')
zip_file = os.path.join(folder, 'bee.zip')

video_file = r'~/Desktop/sample_1280x720_surfing.mkv'

if __name__ == "__main__":
    # download_and_unzip(URL, zip_file, folder)
    
    # source = cv2.VideoCapture(os.path.join(folder, 'bee.mp4'))
    source = cv2.VideoCapture(video_file)
    win_name = 'Video Preview'
    
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while(cv2.waitKey(1) != 27): # 27 is the ASCII code for the ESC key
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyAllWindows()