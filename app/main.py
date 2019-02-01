import os

import cv2
import numpy as np

from app.histogram import generate_video_histogram


def main():
    train_hist_classifier()
    test_hist_classifier()


def train_hist_classifier():
    directory = "../footage/"

    for file in get_video_filenames(directory):
        print("generating histogram for {}".format(file))
        generate_video_histogram(directory, file)
    print("Generated histograms for all files in directory {}".format(directory))


def test_hist_classifier():
    # get histogram for the recorded video to match - todo: calculate the histogram on the go
    directory = "../recordings/"
    histcmp_methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
    hist_recording = {
        'b': np.loadtxt('../histogram_data/recording.mp4/hist-b', dtype=np.float32, unpack=False),
        'g': np.loadtxt('../histogram_data/recording.mp4/hist-g', dtype=np.float32, unpack=False),
        'r': np.loadtxt('../histogram_data/recording.mp4/hist-r', dtype=np.float32, unpack=False)
    }

    # compare recorded video histogram with histogram of each video
    for m in histcmp_methods:
        print("------------------------------------")
        if m == 0:
            print("CORRELATION (highest)")
        elif m == 1:
            print("INTERSECTION (highest)")
        elif m == 2:
            print("CHI SQUARE (lowest)")
        else:
            print("BHATTACHARYYA (lowest)")
        for file in get_video_filenames("../footage/"):
            hist_b = np.loadtxt('../histogram_data/{}/hist-b'.format(file), dtype=np.float32, unpack=False)
            hist_g = np.loadtxt('../histogram_data/{}/hist-g'.format(file), dtype=np.float32, unpack=False)
            hist_r = np.loadtxt('../histogram_data/{}/hist-r'.format(file), dtype=np.float32, unpack=False)
            comparison_b = cv2.compareHist(hist_recording['b'], hist_b, m)
            comparison_g = cv2.compareHist(hist_recording['g'], hist_g, m)
            comparison_r = cv2.compareHist(hist_recording['r'], hist_r, m)
            comparison = (comparison_b + comparison_g + comparison_r) / 3
            print("comparison with {} = {}".format(file, comparison))


def get_video_filenames(directory):
    list_of_videos = list()
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            list_of_videos.append(filename)
        else:
            print("no mp4 files found in directory '{}'".format(directory))
    return list_of_videos


if __name__ == "__main__":
    main()
