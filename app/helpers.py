import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from terminaltables import DoubleTable


def get_video_filenames(directory):
    """
    Returns a list containing all the mp4 files in a directory
    :param directory: the directory containing mp4 files
    :return: list of strings
    """
    list_of_videos = list()
    for filename in os.listdir(directory):
        if filename == ".DS_Store":
            pass  # ignoring .DS_Store file
        elif filename.endswith(".mp4"):
            list_of_videos.append(filename)
        else:
            print("no mp4 files found in directory '{}'".format(directory))
    return list_of_videos


def print_terminal_table(table_data, method_used):
    """
    Prints a table with the results in the terminal.
    :param table_data: the data of the table
    :param method_used: the method used, to print as the table title
    :return: None
    """
    table = DoubleTable(table_data)
    table.title = method_used
    table.inner_heading_row_border = False
    table.inner_row_border = True
    print(table.table)


def print_finished_training_message(answer, model, runtime, accuracy=None):
    """
    Prints a message at the end of the training function.
    :param answer: the matched video name
    :param model: the histogram model used for training
    :param runtime: the time elapsed in seconds
    :param accuracy: the accuracy of the classifier in % (True Positives / Number of Matches)
    :return: None
    """
    print("\n\nGenerated " + "\x1b[1;31m" + "{}".format(model) + "\x1b[0m" + " histograms for all videos")
    if accuracy is not None:
        print("\n\n" + "\x1b[1;31m" + "MATCH FOUND: {}".format(answer) + "\x1b[0m")
    print("\n--- Runtime: {} seconds ---".format(runtime))
    if accuracy is not None:
        print("--- Accuracy: {} % ---".format(round(accuracy * 100, 2)))


def get_video_first_frame(video, path_output_dir, is_query=False, is_result=False):
    """
    Retrieves the first frame from a video and saves it as a PNG.
    :param video: the path to the video
    :param path_output_dir: the directory to save the frame in
    :param is_query: write first frame for query
    :param is_result: write first frame for matched video
    :return: None
    """
    vc = cv2.VideoCapture(video)
    frame_counter = 0
    while vc.isOpened():
        ret, image = vc.read()
        if ret and frame_counter == 0:
            if is_query:
                cv2.imwrite(os.path.join(path_output_dir, "query.png"), image)
            elif is_result:
                cv2.imwrite(os.path.join(path_output_dir, "result.png"), image)
            frame_counter += 1
        else:
            break
    cv2.destroyAllWindows()
    vc.release()


def show_final_match(result_name, query_frame, result_frame, runtime, accuracy):
    """
    Plots the query image and the matched video.
    :param result_name: the name of the matched video
    :param query_frame: the query image
    :param result_frame: the matched video's image
    :param runtime: the time elapsed in seconds
    :param accuracy: the accuracy of the classifier in % (True Positives / Number of Matches)
    :return: None
    """
    query_img = mpimg.imread(query_frame)
    result_img = mpimg.imread(result_frame)
    plt.subplot(2, 1, 1)
    plt.imshow(query_img)
    plt.title("Original Query Video", fontSize=16), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.imshow(result_img)
    plt.title(
        "Match '{}' found in {}s with {}% accuracy".format(result_name, runtime, round(accuracy * 100, 2)),
        fontSize=13)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def display_results_histogram(results_dict):
    """
    Displays the results in the form of a histogram.
    :param results_dict: the histogram with results and the number of matches per video
    :return: None
    """
    plt.bar(list(results_dict.keys()), results_dict.values())
    plt.title("Number of video matches made")
    plt.xlabel("Videos")
    plt.ylabel("Number of matches")
    plt.show()


def get_number_of_frames(vc):
    """
    Retrieves the total number of frames in a video using OpenCV's VideoCapture object cv2.CAP_PROP_FRAME_COUNT
    attribute.
    :param vc: the video capture
    :return: the number of frames in the video capture
    """
    return int(vc.get(cv2.CAP_PROP_FRAME_COUNT))


def get_video_fps(vc):
    """
    Retrieves the frame rate (Frames Per Second) of a video using OpenCV's VideoCapture object cv2.CAP_PROP_FPS
    attribute.
    :param vc: the video capture
    :return: the video capture's FPS
    """
    return round(vc.get(cv2.CAP_PROP_FPS), 2)
