import os

import cv2
import math
from matplotlib import pyplot as plt
import numpy as np


def main():
    colours = ('b', 'g', 'r')
    histograms_dict = {
        'b': list(),
        'g': list(),
        'r': list()
    }
    avg_histogram = np.zeros(shape=(255, 1))  # array to store average histogram values

    directory = "../footage/"
    file_name = "1-waves"
    # file_name = "2-butterfly"
    # filenames = get_video_filenames(directory)
    video_capture = cv2.VideoCapture("{}{}.mp4".format(directory, file_name))
    if not video_capture.isOpened():
        print("Error opening video file")

    # determine which frames to process for histograms
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames_to_process = get_frames_to_process(total_frames, fps)

    # read the video and store the histograms for each frame per color channel in a dict
    frame_counter = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # read capture frame by frame
        if ret:
            frame_counter += 1
            if frame_counter in frames_to_process:
                for i, col in enumerate(colours):
                    histogram = cv2.calcHist([frame], [i], None, [256], [0, 256])
                    histograms_dict[col].append(histogram)
                    # debugging:
                    # print("i: {}, col: {}".format(i, col))
                    # plt.plot(histogram, color=col)
                    # plt.xlim([0, 256])
                # plt.show()

                # user exit on "q" or "Esc" key press
                k = cv2.waitKey(30) & 0xFF
                if k == 25 or k == 27:
                    break
        else:
            break

    # generate a single histogram by averaging all histograms of a video
    for col, hists in histograms_dict.items():
        for i in range(0, 255):  # loop through all bins
            bin_sum = 0

            # get value for each colour histogram in bin i
            for arr_index in range(0, len(hists)):
                bin_value = hists[arr_index].item(i)
                bin_sum += bin_value

            # average all bins values to store in new histogram
            new_bin_value = bin_sum / len(hists)
            avg_histogram[i] = new_bin_value

        np.savetxt('../histogram_data/{}-avg-histogram-{}'.format(file_name, col), avg_histogram, fmt='%d')
        plt.plot(avg_histogram, color=col)
        plt.xlim([0, 256])
    plt.show()

    # tidying up OpenCV video environment
    video_capture.release()
    cv2.destroyAllWindows()


def get_video_filenames(directory):
    list_of_videos = list()
    for filename in os.listdir(""):
        if filename.endswith(".mp4"):
            f = os.path.join(directory, filename)
            list_of_videos.append(f)
        else:
            print("no mp4 files found in directory '{}'".format(directory))
    return list_of_videos


def get_frames_to_process(total_frames, fps):
    frame_ids = list()
    for i in range(1, int(total_frames) + 1, math.ceil(fps)):
        frame_ids.append(i)
    return frame_ids


if __name__ == "__main__":
    main()
