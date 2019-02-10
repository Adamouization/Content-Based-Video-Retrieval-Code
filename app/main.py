import os

import cv2
import numpy as np

from app.histogram import HistogramGenerator


def main():
    train_hist_classifier()
    test_hist_classifier()


def train_hist_classifier():
    """
    Generates an averaged BGR histogram for all the videos in the directory-based database.
    :return: None
    """
    directory = "../footage/"

    for file in get_video_filenames(directory):
        print("generating histogram for {}".format(file))
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_histograms()
    print("Generated histograms for all files in directory {}".format(directory))


def test_hist_classifier():
    """
    Compares the BGR histogram of a mobile-recorded video (of one of the DB videos) and compares it with each of the
    videos' saved BGR histograms using different histogram matching methods such as the Chi-Square or Bhattacharyya
    methods, and prints the results as probabilities.
    :return: None
    """
    directory = "../recordings/"
    file = "recording.mp4"

    # calculate histogram for the recorded video
    print("Please crop the recorded video for the histogram to be generated.")
    histogram_generator = HistogramGenerator(directory, file)
    histogram_generator.generate_recording_video_histograms()

    # variables used for finding the match to the recorded video
    video_match = ""
    video_match_value = 0
    histcmp_methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]

    # get histogram for the recorded video to match - todo: calculate the histogram on the go
    hist_recording = {
        'b': np.loadtxt('../histogram_data/recording.mp4/hist-b', dtype=np.float32, unpack=False),
        'g': np.loadtxt('../histogram_data/recording.mp4/hist-g', dtype=np.float32, unpack=False),
        'r': np.loadtxt('../histogram_data/recording.mp4/hist-r', dtype=np.float32, unpack=False)
    }

    # compare recorded video histogram with histogram of each video
    print("Histogram Comparison Results:")
    for m in histcmp_methods:
        print("------------------------------------")
        if m == 0:
            print("CORRELATION")
        elif m == 1:
            print("INTERSECTION")
        elif m == 2:
            print("CHI SQUARE")
        else:
            print("BHATTACHARYYA")
        for i, file in enumerate(get_video_filenames("../footage/")):
            hist_b = np.loadtxt('../histogram_data/{}/hist-b'.format(file), dtype=np.float32, unpack=False)
            hist_g = np.loadtxt('../histogram_data/{}/hist-g'.format(file), dtype=np.float32, unpack=False)
            hist_r = np.loadtxt('../histogram_data/{}/hist-r'.format(file), dtype=np.float32, unpack=False)
            comparison_b = cv2.compareHist(hist_recording['b'], hist_b, m)
            comparison_g = cv2.compareHist(hist_recording['g'], hist_g, m)
            comparison_r = cv2.compareHist(hist_recording['r'], hist_r, m)
            comparison = (comparison_b + comparison_g + comparison_r) / 3
            print("comparison with {} = {}".format(file, comparison))
            if i == 0:
                video_match = file
                video_match_value = comparison
            else:
                if m in [0, 2] and comparison > video_match_value:
                    video_match = file
                    video_match_value = comparison
                elif m in [1, 3] and comparison < video_match_value:
                    video_match = file
                    video_match_value = comparison
        print("Match found: {}".format(video_match))


def get_video_filenames(directory):
    """
    Returns a list containing all the mp4 files in a directory
    :param directory: the directory containing mp4 files
    :return: list of strings
    """
    list_of_videos = list()
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            list_of_videos.append(filename)
        else:
            print("no mp4 files found in directory '{}'".format(directory))
    return list_of_videos


if __name__ == "__main__":
    main()
