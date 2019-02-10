import math
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np

from app.video_operations import ClickAndDrop


class HistogramGenerator:
    colours = ('b', 'g', 'r')

    def __init__(self, directory, file_name):
        """
        Initialise variables and create a VideoCapture object for a mp4 file
        :param directory: the directory where the video file is located
        :param file_name: the mp4 video file's name
        """
        self.directory = directory
        self.file_name = file_name

        # start capturing video
        self.video_capture = cv2.VideoCapture("{}{}".format(self.directory, self.file_name))
        self.check_video_capture()

        # read the video and store the histograms for each frame per color channel in a dict
        self.histograms_dict = {
            'b': list(),
            'g': list(),
            'r': list()
        }

    def generate_video_histograms(self):
        """
        Generates multiple BGR histograms (one every second) for the video.
        :return: None
        """
        # determine which frames to process for histograms
        frames_to_process = _get_frames_to_process(self.video_capture)

        frame_counter = 0  # keep track of current frame ID to know to process it or not
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # read capture frame by frame
            if ret:
                frame_counter += 1
                if frame_counter in frames_to_process:
                    for i, col in enumerate(self.colours):
                        histogram = cv2.calcHist([frame], [i], None, [256], [0, 256])
                        histogram = cv2.normalize(histogram, histogram)
                        self.histograms_dict[col].append(histogram)
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
        self.generate_and_store_average_histogram()
        self.destroy_video_capture()

    def generate_recording_video_histograms(self):
        """

        :return:
        """
        # determine which frames to process for histograms
        frames_to_process = _get_frames_to_process(self.video_capture)

        reference_points = list()
        frame_counter = 0  # keep track of current frame ID to know to process it or not
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # read capture frame by frame
            if ret:
                if frame_counter == 0:
                    cad = ClickAndDrop(frame)
                    # debugging: uncomment to show the cropped frame
                    # roi_frame = cad.get_roi()
                    # cv2.imshow('Selected ROI', roi_frame)
                    # cv2.waitKey(0)
                    reference_points = cad.get_reference_points()
                frame_counter += 1
                if frame_counter in frames_to_process:
                    for i, col in enumerate(self.colours):
                        if len(reference_points) == 2:
                            roi = frame[reference_points[0][1]:reference_points[1][1],
                                        reference_points[0][0]:reference_points[1][0]]
                            histogram = cv2.calcHist([roi], [i], None, [256], [0, 256])
                            histogram = cv2.normalize(histogram, histogram)
                            self.histograms_dict[col].append(histogram)
                            # debugging: uncomment to show individual BGR histogram plots
                            # print("i: {}, col: {}".format(i, col))
                            # plt.plot(histogram, color=col)
                            # plt.xlim([0, 256])
                        else:
                            print("Error while cropping the recording")
                            exit(0)
                    # plt.show()

                    # user exit on "q" or "Esc" key press
                    k = cv2.waitKey(30) & 0xFF
                    if k == 25 or k == 27:
                        break
            else:
                break
        self.generate_and_store_average_histogram()
        self.destroy_video_capture()

    def generate_and_store_average_histogram(self):
        """
        Generates a single BGR histogram by averaging all histograms of a video before writing the results to a txt
        file.
        :return: None
        """
        avg_histogram = np.zeros(shape=(255, 1))  # array to store average histogram values
        for col, hists in self.histograms_dict.items():
            for i in range(0, 255):  # loop through all bins
                bin_sum = 0

                # get value for each colour histogram in bin i
                for arr_index in range(0, len(hists)):
                    bin_value = hists[arr_index].item(i)
                    bin_sum += bin_value

                # average all bins values to store in new histogram
                new_bin_value = bin_sum / len(hists)
                avg_histogram[i] = new_bin_value

            if not os.path.exists("../histogram_data/{}/".format(self.file_name)):
                os.makedirs("../histogram_data/{}/".format(self.file_name))
            np.savetxt("../histogram_data/{}/hist-{}".format(self.file_name, col), avg_histogram, fmt='%f')
            plt.plot(avg_histogram, color=col)
            plt.xlim([0, 256])
        plt.title('{}'.format(self.file_name))
        plt.show()

    def check_video_capture(self):
        """
        Checks if the VideoCapture object was correctly created.
        :return: None
        """
        if not self.video_capture.isOpened():
            print("Error opening video file")

    def destroy_video_capture(self):
        """
        Tidying up the OpenCV environment and the video capture
        :return: None
        """
        self.video_capture.release()
        cv2.destroyAllWindows()

    def get_video_capture(self):
        """
        Returns the full VideoCapture object.
        :return: the VideoCapture object
        """
        return self.video_capture


def _get_frames_to_process(vc):
    """
    Returns the IDs of the frames to calculate a BGR histogram for.
    :param vc: the VideoCapture object to process
    :return: a list of integers representing the frames to process
    """
    frame_ids = list()
    total_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    for i in range(1, int(total_frames) + 1, math.ceil(fps)):
        frame_ids.append(i)
    return frame_ids
