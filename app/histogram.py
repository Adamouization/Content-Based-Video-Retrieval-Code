import math
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np


colours = ('b', 'g', 'r')


def generate_video_histogram(directory, file_name):
    """
    Creates a VideoCapture object for a mp4 file and generates an averaged and normalized BGR histogram for the video
    before writing the results to a txt file.
    :param directory: the directory where the video file is located
    :param file_name: the mp4 video file's name
    :return: None
    """
    # start capturing video
    video_capture = cv2.VideoCapture("{}{}".format(directory, file_name))
    if not video_capture.isOpened():
        print("Error opening video file")

    # determine which frames to process for histograms
    frames_to_process = _get_frames_to_process(video_capture)

    # read the video and store the histograms for each frame per color channel in a dict
    histograms_dict = {
        'b': list(),
        'g': list(),
        'r': list()
    }
    frame_counter = 0  # keep track of current frame ID to know to process it or not
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # read capture frame by frame
        if ret:
            frame_counter += 1
            if frame_counter in frames_to_process:
                for i, col in enumerate(colours):
                    histogram = cv2.calcHist([frame], [i], None, [256], [0, 256])
                    histogram = cv2.normalize(histogram, histogram)
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
    avg_histogram = np.zeros(shape=(255, 1))  # array to store average histogram values
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

        if not os.path.exists("../histogram_data/{}/".format(file_name)):
            os.makedirs("../histogram_data/{}/".format(file_name))
        np.savetxt("../histogram_data/{}/hist-{}".format(file_name, col), avg_histogram, fmt='%f')
        plt.plot(avg_histogram, color=col)
        plt.xlim([0, 256])
    plt.show()

    # tidying up OpenCV video environment
    video_capture.release()
    cv2.destroyAllWindows()


def _get_frames_to_process(video_capture):
    """
    Returns the IDs of the frames to calculate a BGR histogram for.
    :param video_capture: the VideoCapture object to process
    :return: a list of integers representing the frames to process
    """
    frame_ids = list()
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    for i in range(1, int(total_frames) + 1, math.ceil(fps)):
        frame_ids.append(i)
    return frame_ids
