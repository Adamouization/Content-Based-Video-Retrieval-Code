import argparse
from collections import Counter
import time

from pyspin.spin import make_spin, Spin2

from app.helpers import get_video_filenames, print_finished_training_message
from app.histogram import HistogramGenerator
import app.config as settings


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
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Specify whether you want to print additional logs for debugging purposes.")
    args = parser.parse_args()
    settings.debug = args.debug
    settings.mode = args.mode
    settings.model = args.model

    if settings.mode == 'train':
        train_hist_classifier()
    elif settings.mode == 'test':
        test_hist_classifier()
    elif settings.mode == 'segment':
        segment_video()
    else:
        print("Wrong mode chosen. Choose from the following options: 'train', 'test' or 'segment'.")
        exit(0)


@make_spin(Spin2, "Generating histograms...".format(settings.model))
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
        if settings.model == "gray":
            histogram_generator = HistogramGenerator(directory, file)
            histogram_generator.generate_video_grayscale_histogram()
        elif settings.model == "rgb":
            histogram_generator = HistogramGenerator(directory, file)
            histogram_generator.generate_video_rgb_histogram()
        elif settings.model == "hsv":
            histogram_generator = HistogramGenerator(directory, file)
            histogram_generator.generate_video_hsv_histogram()
        else:
            histogram_generator_gray = HistogramGenerator(directory, file)
            histogram_generator_gray.generate_video_grayscale_histogram()
            histogram_generator_rgb = HistogramGenerator(directory, file)
            histogram_generator_rgb.generate_video_rgb_histogram()
            histogram_generator_hsv = HistogramGenerator(directory, file)
            histogram_generator_hsv.generate_video_hsv_histogram()
    runtime = round(time.time() - start_time, 2)
    print_finished_training_message(settings.model, directory, runtime)


def test_hist_classifier():
    """
    Prompts the user to crop the recorded video before generating an averaged BGR histogram and comparing it with the
    other averaged histograms for matching.
    :return: None
    """
    directory = "../recordings/"
    recordings = ["recording1.mp4", "recording2.mp4", "recording3.mp4"]  # 1: cloud-sky, 2: seal, 3: butterfly
    file = recordings[0]

    print("\nPlease crop the recorded video for the histogram to be generated.")

    if settings.model == "gray":
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_grayscale_histogram(is_query=True)
        histogram_generator.match_histograms()
    elif settings.model == "rgb":
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_rgb_histogram(is_query=True)
        histogram_generator.match_histograms()
    elif settings.model == "hsv":
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_hsv_histogram(is_query=True)
        histogram_generator.match_histograms()
    else:
        # calculate query histogram
        # gray scale
        histogram_generator_gray = HistogramGenerator(directory, file)
        histogram_generator_gray.generate_video_grayscale_histogram(is_query=True)
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
        gray_results = histogram_generator_gray.get_results_array()
        histogram_generator_rgb.match_histograms(cur_all_model='rgb')
        rgb_results = histogram_generator_rgb.get_results_array()
        histogram_generator_hsv.match_histograms(cur_all_model='hsv')
        hsv_results = histogram_generator_hsv.get_results_array()

        # combine matches from all 3 histogram models to output one final result
        all_results = gray_results + rgb_results + hsv_results
        results_count = Counter(all_results)
        # print(results_count)
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
        runtime = round(time.time() - start_time, 2)
        print_finished_training_message(final_result_name, settings.model, runtime)


def segment_video():
    directory = "../recordings/"
    video = "scene-segmentation.mp4"

    shot_boundary_detector = HistogramGenerator(directory, video)
    shot_boundary_detector.rgb_histogram_shot_boundary_detection()
    print("Finished detecting shot boundaries for {}.".format(video))


if __name__ == "__main__":
    main()
