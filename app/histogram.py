import math
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np


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

        # array to store average histogram values
        self.avg_histogram = np.zeros(shape=(255, 1))

    def generate_and_store_video_histogram(self):
        """
        Generates an averaged and normalized BGR histogram for the video before writing the results to a txt file.
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

        # generate a single histogram by averaging all histograms of a video
        for col, hists in self.histograms_dict.items():
            for i in range(0, 255):  # loop through all bins
                bin_sum = 0

                # get value for each colour histogram in bin i
                for arr_index in range(0, len(hists)):
                    bin_value = hists[arr_index].item(i)
                    bin_sum += bin_value

                # average all bins values to store in new histogram
                new_bin_value = bin_sum / len(hists)
                self.avg_histogram[i] = new_bin_value

            if not os.path.exists("../histogram_data/{}/".format(self.file_name)):
                os.makedirs("../histogram_data/{}/".format(self.file_name))
            np.savetxt("../histogram_data/{}/hist-{}".format(self.file_name, col), self.avg_histogram, fmt='%f')
            plt.plot(self.avg_histogram, color=col)
            plt.xlim([0, 256])
        plt.title('{}'.format(self.file_name))
        plt.show()
        self.destroy_video_capture()

    def generate_and_store_recording_video_histogram(self):
        # determine which frames to process for histograms
        frames_to_process = _get_frames_to_process(self.video_capture)

        frame_counter = 0  # keep track of current frame ID to know to process it or not
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()  # read capture frame by frame
            if ret:
                if frame_counter == 0:
                    cad = ClickAndDrop(frame)
                    # roi = cad.get_roi()
                    # cv2.imshow('Selected ROI', roi)
                    # cv2.waitKey(0)
                    # todo use these reference points to ignore pixels outside when generating histogram
                    refpt = cad.get_refpt()
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


class ClickAndDrop:
    """
    Class for selecting a region of interest on a frame by click and dropping the mouse over the desired area, and
    cropping that frame to include the pixels inside the ROI only.
    """
    frame_size = (1280, 720)

    def __init__(self, thumbnail):
        self.thumbnail = thumbnail

        self.refPt = list()
        self.cropping = False

        # load the image, clone it, and setup the mouse callback function
        self.image = cv2.imread("/Users/ajaamour/Projects/Content-Based-Video-Retrieval-Code/recordings/test.png")
        self.image = cv2.resize(self.image, self.frame_size, interpolation=cv2.INTER_AREA)
        clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if len(self.refPt) == 2:
            self.roi = clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
            # cv2.imshow("ROI", self.roi)
            # cv2.waitKey(0)

        # close all open windows
        cv2.destroyAllWindows()

    def click_and_crop(self, event, x, y, flags, param):
        """
        Callback function allowing a user to manually crop an image.

        Function written by Adrian Rosebrock
        Link: https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

        :param event: one of the MouseEventTypes constants
        :param x: the x-coordinate of the mouse event
        :param y: the y-coordinate of the mouse event
        :param flags: one of the MouseEventFlags constants
        :param param: optional parameters
        :return: None
        """
        # grab references to the global variables
        # global refPt, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", self.image)

    def get_roi(self):
        """
        Return the selected region of interest of the frame.
        :return: roi: the user-cropped region of the image
        """
        return self.roi

    def get_refpt(self):
        """
        Returns the ROI's reference points coordinates
        :return:
        """
        return self.refPt


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
