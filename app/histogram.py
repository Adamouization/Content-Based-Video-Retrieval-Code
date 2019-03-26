import csv
import math
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
from terminaltables import DoubleTable

from app.helpers import get_video_filenames
import app.config as config
from app.video_operations import ClickAndDrop


class HistogramGenerator:
    colours = ('b', 'g', 'r')
    bins = (8, 12, 3)  # 8 hue bins, 12 saturation bins, 3 value bins
    histcmp_methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA,
                       cv2.HISTCMP_CHISQR_ALT, cv2.HISTCMP_KL_DIV]

    def __init__(self, directory, file_name):
        """
        Initialise variables and create a VideoCapture object for a mp4 file.
        :param directory: the directory where the video file is located
        :param file_name: the mp4 video file's name
        """
        self.directory = directory
        self.file_name = file_name

        # start capturing video
        self.video_capture = cv2.VideoCapture("{}{}".format(self.directory, self.file_name))
        self.check_video_capture()

        # read the video and store the histograms for each frame per color channel in a dict
        self.histograms_gray_dict = list()
        self.histograms_rgb_dict = {
            'b': list(),
            'g': list(),
            'r': list()
        }
        self.histograms_hsv_dict = list()

    def generate_video_rgb_histogram(self, is_query=False):
        """
        Generates multiple normalized histograms (one every second) for a video.
        :param is_query: boolean specifying if the input video is the query video (to select ROI)
        :return: None
        """
        # determine which frames to process for histograms
        frames_to_process = _get_frames_to_process(self.video_capture)

        reference_points = list()
        frame_counter = 0  # keep track of current frame ID to know to process it or not
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # read capture frame by frame
            if ret:
                if is_query and frame_counter == 0:
                    cad = ClickAndDrop(frame)
                    if config.debug:  # show the cropped region of interest
                        roi_frame = cad.get_roi()
                        cv2.imshow('Selected ROI', roi_frame)
                        cv2.waitKey(0)
                    reference_points = cad.get_reference_points()
                frame_counter += 1
                if frame_counter in frames_to_process:
                    for i, col in enumerate(self.colours):
                        if is_query and len(reference_points) == 2:
                            roi = frame[reference_points[0][1]:reference_points[1][1],
                                        reference_points[0][0]:reference_points[1][0]]
                            histogram = cv2.calcHist([roi], [i], None, [256], [0, 256])
                        else:
                            histogram = cv2.calcHist([frame], [i], None, [256], [0, 256])
                        histogram = cv2.normalize(histogram, histogram)
                        self.histograms_rgb_dict[col].append(histogram)
                        if config.debug:  # show individual BGR histogram plots
                            print("i: {}, col: {}".format(i, col))
                            plt.plot(histogram, color=col)
                            plt.xlim([0, 256])
                    if config.debug:
                        plt.show()

                    # user exit on "q" or "Esc" key press
                    k = cv2.waitKey(30) & 0xFF
                    if k == 25 or k == 27:
                        break
            else:
                break
        self.generate_and_store_average_rgb_histogram()
        self.destroy_video_capture()

    def generate_video_grayscale_histogram(self, is_query=False):
        """
        Generates multiple normalized grayscale histograms (one every second) for a video.
        :param is_query: boolean specifying if the input video is the query video (to select ROI)
        :return: None
        """
        # determine which frames to process for histograms
        frames_to_process = _get_frames_to_process(self.video_capture)

        reference_points = list()
        frame_counter = 0  # keep track of current frame ID to know to process it or not
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # read capture frame by frame
            if ret:
                if is_query and frame_counter == 0:
                    cad = ClickAndDrop(frame)
                    if config.debug:  # show the cropped region of interest
                        roi_frame = cad.get_roi()
                        cv2.imshow("Selected ROI", roi_frame)
                        cv2.waitKey(0)
                    reference_points = cad.get_reference_points()
                frame_counter += 1
                if frame_counter in frames_to_process:
                    if is_query and len(reference_points) == 2:
                        roi = frame[reference_points[0][1]:reference_points[1][1],
                                    reference_points[0][0]:reference_points[1][0]]
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        histogram = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
                    else:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
                    histogram = cv2.normalize(histogram, histogram)
                    self.histograms_gray_dict.append(histogram)
                    if config.debug:  # show individual grayscale histogram plots
                        plt.figure()
                        plt.title("{} frame {}".format(self.file_name, frame_counter))
                        plt.xlabel("Bins")
                        plt.plot(histogram)
                        plt.xlim([0, 256])
                        plt.show()

                    # user exit on "q" or "Esc" key press
                    k = cv2.waitKey(30) & 0xFF
                    if k == 25 or k == 27:
                        break
            else:
                break
        self.generate_and_store_average_grayscale_histogram()
        self.destroy_video_capture()

    def generate_video_hsv_histogram(self, is_query=False):
        # determine which frames to process for histograms
        frames_to_process = _get_frames_to_process(self.video_capture)

        reference_points = list()
        frame_counter = 0  # keep track of current frame ID to know to process it or not
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # read capture frame by frame
            if ret:
                if is_query and frame_counter == 0:
                    cad = ClickAndDrop(frame)
                    if config.debug:  # show the cropped region of interest
                        roi_frame = cad.get_roi()
                        cv2.imshow("Selected ROI", roi_frame)
                        cv2.waitKey(0)
                    reference_points = cad.get_reference_points()
                frame_counter += 1
                if frame_counter in frames_to_process:
                    if is_query and len(reference_points) == 2:
                        roi = frame[reference_points[0][1]:reference_points[1][1],
                                    reference_points[0][0]:reference_points[1][0]]
                        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        histogram = cv2.calcHist([roi_hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
                    else:
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        histogram = cv2.calcHist([hsv_frame], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
                    histogram = cv2.normalize(histogram, histogram)
                    self.histograms_hsv_dict.append(histogram)
                    if config.debug:  # show individual HSV histogram plots
                        plt.imshow(histogram)
                        plt.title("{} frame {}".format(self.file_name, frame_counter))
                        plt.show()

                    # user exit on "q" or "Esc" key press
                    k = cv2.waitKey(30) & 0xFF
                    if k == 25 or k == 27:
                        break
            else:
                break
        self.generate_and_store_average_hsv_histogram()
        self.destroy_video_capture()

    def generate_and_store_average_rgb_histogram(self):
        """
        Generates a single BGR histogram by averaging all histograms of a video before writing the results to a txt
        file.
        :return: None
        """
        avg_histogram = np.zeros(shape=(255, 1))  # array to store average histogram values
        for col, hists in self.histograms_rgb_dict.items():
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
        plt.title("RGB histogram for '{}'".format(self.file_name))
        plt.xlabel("Bins")
        plt.show()

    def generate_and_store_average_grayscale_histogram(self):
        """
        Generates a single BGR histogram by averaging all histograms of a video before writing the results to a txt
        file.
        :return: None
        """
        avg_histogram = np.zeros(shape=(255, 1))  # array to store average histogram values

        col = "gray"
        hist = self.histograms_gray_dict

        for i in range(0, 255):  # loop through all bins
            bin_sum = 0

            # get value for each colour histogram in bin i
            for arr_index in range(0, len(hist)):
                bin_value = hist[arr_index].item(i)
                bin_sum += bin_value

            # average all bins values to store in new histogram
            new_bin_value = bin_sum / len(hist)
            avg_histogram[i] = new_bin_value

        if not os.path.exists("../histogram_data/{}/".format(self.file_name)):
            os.makedirs("../histogram_data/{}/".format(self.file_name))
        np.savetxt("../histogram_data/{}/hist-{}".format(self.file_name, col), avg_histogram, fmt='%f')
        plt.plot(avg_histogram, color=col)
        plt.xlim([0, 256])
        plt.title("Grayscale histogram for '{}'".format(self.file_name))
        plt.xlabel("Bins")
        plt.show()

    def generate_and_store_average_hsv_histogram(self):
        """
        Generates a single BGR histogram by averaging all histograms of a video before writing the results to a txt
        file.
        :return: None
        """
        avg_histogram = np.zeros(shape=(255, 1))  # array to store average histogram values

        col = "hsv"
        hist = self.histograms_hsv_dict

        # todo: calculate average histogram

        for h in range(0, self.bins[0]):  # loop through all bins
            for s in range(0, self.bins[1]):
                for v in range(0, self.bins[2]):
                    print(h, s, v)
                    bin_sum = 0

                    # get value for each colour histogram in bin i
                    for arr_index in range(0, len(hist)):
                        bin_value = hist[arr_index].item(h)
                        bin_sum += bin_value

                    # average all bins values to store in new histogram
                    new_bin_value = bin_sum / len(hist)
                    avg_histogram[i] = new_bin_value

        if not os.path.exists("../histogram_data/{}/".format(self.file_name)):
            os.makedirs("../histogram_data/{}/".format(self.file_name))
        np.savetxt("../histogram_data/{}/hist-{}".format(self.file_name, col), avg_histogram, fmt='%f')
        plt.imshow(avg_histogram)
        plt.title("HSV histogram for '{}'".format(self.file_name))
        plt.show()

    def match_histograms(self):
        """
        Compares the BGR histogram of the recorded video and compares it with each of the saved average BGR histograms
        using different histogram matching methods such as the Chi-Square or Bhattacharyya methods, and prints the
        results as probabilities in a CLI table and writes the data to a CSV file.
        :return: None
        """
        # variables used for finding the match to the recorded video
        video_match = ""
        video_match_value = 0

        # get histogram for the recorded video to match - todo: calculate the histogram on the go
        hist_recording = dict()
        if config.model == "gray":
            hist_recording = {
                'gray': np.loadtxt("../histogram_data/{}/hist-gray".format(self.file_name), dtype=np.float32, unpack=False),
            }
        elif config.model == "rgb":
            hist_recording = {
                'b': np.loadtxt("../histogram_data/{}/hist-b".format(self.file_name), dtype=np.float32, unpack=False),
                'g': np.loadtxt("../histogram_data/{}/hist-g".format(self.file_name), dtype=np.float32, unpack=False),
                'r': np.loadtxt("../histogram_data/{}/hist-r".format(self.file_name), dtype=np.float32, unpack=False)
            }

        # compare recorded video histogram with histogram of each video
        print("\n{} Histogram Comparison Results:\n".format(_get_chosen_model_string()))
        method = ""
        for m in self.histcmp_methods:

            if m == 0:
                method = "CORRELATION"
            elif m == 1:
                method = "INTERSECTION"
            elif m == 2:
                method = "CHI SQUARE"
            elif m == 3:
                method = "BHATTACHARYYA"
            elif m == 4:
                method = "ALTERNATIVE CHI-SQUARE"
            elif m == 5:
                method = "KULLBACK-LEIBLER DIVERGENCE"

            # CSV file to write data to for each method
            csv_file = open('../results/csv/{}-{}.csv'.format(config.model, method), 'w')
            with csv_file:
                field_names = ['video', 'score']
                writer = csv.DictWriter(csv_file, fieldnames=field_names)
                writer.writeheader()

                table_data = list()
                for i, file in enumerate(get_video_filenames("../footage/")):
                    comparison = 0
                    if config.model == "gray":
                        hist_gray = np.loadtxt("../histogram_data/{}/hist-gray".format(file), dtype=np.float32, unpack=False)
                        comparison = cv2.compareHist(hist_recording['gray'], hist_gray, m)
                    elif config.model == "rgb":
                        hist_b = np.loadtxt("../histogram_data/{}/hist-b".format(file), dtype=np.float32, unpack=False)
                        hist_g = np.loadtxt("../histogram_data/{}/hist-g".format(file), dtype=np.float32, unpack=False)
                        hist_r = np.loadtxt("../histogram_data/{}/hist-r".format(file), dtype=np.float32, unpack=False)
                        comparison_b = cv2.compareHist(hist_recording['b'], hist_b, m)
                        comparison_g = cv2.compareHist(hist_recording['g'], hist_g, m)
                        comparison_r = cv2.compareHist(hist_recording['r'], hist_r, m)
                        comparison = (comparison_b + comparison_g + comparison_r) / 3

                    # append data to table
                    table_data.append([file, round(comparison, 5)])

                    # write data to CSV file
                    writer.writerow({'video': file, 'score': round(comparison, 5)})

                    if i == 0:
                        video_match = file
                        video_match_value = comparison
                    else:
                        if m in [0, 2] and comparison > video_match_value:
                            video_match = file
                            video_match_value = comparison
                        elif m in [1, 3, 4, 5] and comparison < video_match_value:
                            video_match = file
                            video_match_value = comparison

                table = DoubleTable(table_data)
                table.title = method
                table.inner_heading_row_border = False
                table.inner_row_border = True
                print(table.table)
                print("Match found: " + "\x1b[1;31m" + video_match + "\x1b[0m" + "\n\n")

    def rgb_histogram_shot_boundary_detection(self):
        """
        Compares consecutive frames' RGB histograms using the Kullback-Leibler Divergence metric. If the metric is
        bigger than the specified threshold, then a shot boundary has been detected.
        :return: None
        """
        threshold = 10
        x_axis = list()
        y_axis = list()
        is_under_threshold = True

        ret, frame = self.video_capture.read()  # get initial frame

        frame_counter = 0  # keep track of current frame ID to locate shot boundaries
        while self.video_capture.isOpened():
            prev_frame = frame[:]  # previous frame
            ret, frame = self.video_capture.read()  # read capture frame by frame

            if ret:
                frame_counter += 1
                cur_rgb_hist = {
                    'b': list(),
                    'g': list(),
                    'r': list()
                }
                prev_rgb_hist = {
                    'b': list(),
                    'g': list(),
                    'r': list()
                }
                for i, col in enumerate(self.colours):
                    # calculate RGB histograms
                    cur_frame_hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
                    prev_frame_hist = cv2.calcHist([prev_frame], [i], None, [256], [0, 256])

                    # normalize histograms
                    cur_frame_hist = cv2.normalize(cur_frame_hist, cur_frame_hist)
                    prev_frame_hist = cv2.normalize(prev_frame_hist, prev_frame_hist)

                    # save histograms in dict
                    cur_rgb_hist[col].append(cur_frame_hist)
                    prev_rgb_hist[col].append(prev_frame_hist)

                # calculate Kullback-Leibler Divergence between consecutive frames
                comparison_r = cv2.compareHist(prev_rgb_hist['r'][0], cur_rgb_hist['r'][0], cv2.HISTCMP_KL_DIV)
                comparison_g = cv2.compareHist(prev_rgb_hist['g'][0], cur_rgb_hist['g'][0], cv2.HISTCMP_KL_DIV)
                comparison_b = cv2.compareHist(prev_rgb_hist['b'][0], cur_rgb_hist['b'][0], cv2.HISTCMP_KL_DIV)
                comparison = (comparison_b + comparison_g + comparison_r) / 3

                # append data to lists for plot
                x_axis.append(frame_counter)
                y_axis.append(comparison)

                if comparison > threshold and is_under_threshold:
                    is_under_threshold = False
                    print("Scene Change detected at Frame {}".format(frame_counter))
                elif comparison < threshold:
                    is_under_threshold = True

            else:
                break

        # Plot results
        plt.plot(x_axis, y_axis)
        plt.plot(x_axis, np.full(frame_counter, 10))
        plt.title("Kullback-Leibler Divergence Between Consecutive Frame RGB Histogram")
        plt.xlabel("Frame")
        plt.ylabel("KL Divergence")
        plt.show()

        self.destroy_video_capture()

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


def _get_chosen_model_string():
    """
    Returns the Histogram Model chosen for the matching process.
    :return: a string representing the chosen histogram model
    """
    if config.model == "gray":
        return "Grayscale"
    elif config.model == "rgb":
        return "RGB"
