import argparse
from collections import Counter
import time

from pyspin.spin import make_spin, Spin2

import app.config as config
from app.helpers import display_results_histogram, get_number_of_frames, get_video_filenames, get_video_fps, \
    get_video_first_frame, print_finished_training_message, terminal_yes_no_question, show_final_match, \
    video_file_already_stabilised
from app.histogram import HistogramGenerator
from app.video_operations import VideoStabilizer


def main():
    """
    Program entry point. Parses command line input.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="The histogram model to use. Choose from the following options: 'rgb', 'hsv' or 'gray'. "
                             "Leave empty to train using all 3 histogram models.")
    parser.add_argument("--mode",
                        required=True,
                        help="The mode to run the code in. Choose from the following options: 'train', 'test' or "
                             "'segment'.")
    parser.add_argument("--showhists",
                        action="store_true",
                        help="Specify whether you want to display each generated histogram.")
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Specify whether you want to print additional logs for debugging purposes.")
    args = parser.parse_args()
    config.debug = args.debug
    config.mode = args.mode
    config.show_histograms = args.showhists
    config.model = args.model

    if config.mode == "train":
        train_hist_classifier()
    elif config.mode == "test":
        test_hist_classifier()
    elif config.mode == "segment":
        segment_video()
    else:
        print("Wrong mode chosen. Choose from the following options: 'train', 'test' or 'segment'.")
        exit(0)


@make_spin(Spin2, "Generating histograms...".format(config.model))
def train_hist_classifier():
    """
    Generates an averaged BGR histogram for all the videos in the directory-based database.
    :return: None
    """
    directory = "../footage/"
    files = get_video_filenames(directory)

    # start measuring runtime
    start_time = time.time()

    for file in files:
        if config.model == "gray":
            histogram_generator = HistogramGenerator(directory, file)
            histogram_generator.generate_video_greyscale_histogram()
        elif config.model == "rgb":
            histogram_generator = HistogramGenerator(directory, file)
            histogram_generator.generate_video_rgb_histogram()
        elif config.model == "hsv":
            histogram_generator = HistogramGenerator(directory, file)
            histogram_generator.generate_video_hsv_histogram()
        else:
            histogram_generator_gray = HistogramGenerator(directory, file)
            histogram_generator_gray.generate_video_greyscale_histogram()
            histogram_generator_rgb = HistogramGenerator(directory, file)
            histogram_generator_rgb.generate_video_rgb_histogram()
            histogram_generator_hsv = HistogramGenerator(directory, file)
            histogram_generator_hsv.generate_video_hsv_histogram()
    runtime = round(time.time() - start_time, 2)
    print_finished_training_message(config.model, directory, runtime)


def test_hist_classifier():
    """
    Prompts the user to crop the recorded video before generating an averaged BGR histogram and comparing it with the
    other averaged histograms for matching.
    :return: None
    """
    directory = "../recordings/"
    recordings = ["recording1.mp4", "recording2.mp4", "recording3.mp4", "recording4.mp4", "recording5.mp4"]
    file = recordings[2]  # 0: cloudy-sky, 1: seal, 2: butterfly (skewed), 3: wind-turbine, 4: butterfly

    # ask user to stabilise the input query video or not
    is_stabilise_video = terminal_yes_no_question("Do you wish to stabilise the recorded query video?")
    stable_filename = "stable-" + file[:-4] + ".avi"  # the stable version of the video
    # yes: stabilise the video and use the stable .avi version
    if is_stabilise_video:
        if not video_file_already_stabilised(directory + stable_filename):
            VideoStabilizer(directory, "{}".format(file))
        file = stable_filename
    # no: check if a version of the stabilised video doesn't already exist - use it if it does
    else:
        if video_file_already_stabilised(directory + stable_filename):
            file = stable_filename

    print("\nUsing query: '{}'".format(file))
    print("\nPlease crop the recorded query video for the signature to be generated.")

    if config.model == "gray":
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_greyscale_histogram(is_query=True)
        histogram_generator.match_histograms()
    elif config.model == "rgb":
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_rgb_histogram(is_query=True)
        histogram_generator.match_histograms()
    elif config.model == "hsv":
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_hsv_histogram(is_query=True)
        histogram_generator.match_histograms()
    else:
        # calculate query histogram
        # gray scale
        histogram_generator_gray = HistogramGenerator(directory, file)
        histogram_generator_gray.generate_video_greyscale_histogram(is_query=True)
        cur_reference_points = histogram_generator_gray.get_current_reference_points()
        # RGB
        histogram_generator_rgb = HistogramGenerator(directory, file)
        histogram_generator_rgb.generate_video_rgb_histogram(is_query=True, cur_ref_points=cur_reference_points)
        # HSV
        histogram_generator_hsv = HistogramGenerator(directory, file)
        histogram_generator_hsv.generate_video_hsv_histogram(is_query=True, cur_ref_points=cur_reference_points)

        # start measuring runtime
        start_time = time.time()

        # calculate distances between query and DB histograms
        histogram_generator_gray.match_histograms(cur_all_model='gray')
        histogram_generator_rgb.match_histograms(cur_all_model='rgb')
        histogram_generator_hsv.match_histograms(cur_all_model='hsv')

        # combine matches from all 3 histogram models to output one final result
        all_results = histogram_generator_hsv.get_results_array()
        results_count = Counter(all_results)
        display_results_histogram(results_count)
        print("Matches made: {}".format(results_count))

        # find best result
        final_result_name = ""
        final_result_count = 0
        for i, r in enumerate(results_count):
            if i == 0:
                final_result_name = r
                final_result_count = results_count[r]
            else:
                if results_count[r] > final_result_count:
                    final_result_name = r
                    final_result_count = results_count[r]

        # print results
        runtime = round(time.time() - start_time, 2)
        accuracy = final_result_count / len(all_results)
        get_video_first_frame(directory + file, "../results", is_query=True)
        get_video_first_frame("../footage/{}".format(final_result_name), "../results", is_result=True)
        show_final_match(final_result_name, "../results/query.png", "../results/result.png", runtime, accuracy)
        print_finished_training_message(final_result_name, config.model, runtime, accuracy)


def segment_video():
    directory = "../recordings/"
    video = "scene-segmentation.mp4"
    # directory = "/Volumes/ADAM2/"
    # movies = ["Inception (2010).mp4"]
    # video = movies[0]

    shot_boundary_detector = HistogramGenerator(directory, video)
    video_capture = shot_boundary_detector.get_video_capture()
    frame_count = get_number_of_frames(vc=video_capture)
    fps = get_video_fps(vc=video_capture)
    print("Total Frames: {}".format(frame_count))
    print("FPS: {}\n".format(fps))

    # start measuring runtime
    start_time = time.time()

    # start processing video for shout boundary detection
    print("Starting to process video for shot boundary detection...")
    shot_boundary_detector.rgb_histogram_shot_boundary_detection(threshold=10)

    # print final results
    runtime = round(time.time() - start_time, 2)
    print("--- Number of frames in video: {} ---".format(frame_count))
    print("--- Runtime: {} seconds ---".format(runtime))


if __name__ == "__main__":
    main()
