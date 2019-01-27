import cv2
from matplotlib import pyplot as plt
import numpy as np


def main():
    frame_count = 0
    color = ('b', 'g', 'r')

    video_capture = cv2.VideoCapture('../footage/test_video.mp4')
    if not video_capture.isOpened():
        print("Error opening video file")

    while video_capture.isOpened():
        ret, frame = video_capture.read()  # read capture frame by frame
        if ret:
            frame_count = frame_count + 1
            if frame_count in [1, 5, 9]:  # todo - replace with some logic to only calculate specific frames
                for i, col in enumerate(color):
                    print("i: {}, col: {}".format(i, col))
                    histogram = cv2.calcHist([frame], [i], None, [256], [0, 256])
                    plt.plot(histogram, color=col)
                    plt.xlim([0, 256])
                plt.show()

                # user exit on "q" or "Esc" key press
                k = cv2.waitKey(30) & 0xFF
                if k == 25 or k == 27:
                    break
        else:
            break

    # tidying up
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
