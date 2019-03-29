import argparse

from pyspin.spin import make_spin, Spin2

from app.helpers import get_video_filenames
from app.histogram import HistogramGenerator
import app.config as settings


def main():
    """
    Program entry point. Parses command line input.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="The histogram model to use. Choose from the following options: 'rgb', 'hsv' or 'gray'.")
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

    for file in files:
        histogram_generator = HistogramGenerator(directory, file)
        if settings.model == "gray":
            histogram_generator.generate_video_grayscale_histogram()
        elif settings.model == "rgb":
            histogram_generator.generate_video_rgb_histogram()
        elif settings.model == "hsv":
            histogram_generator.generate_video_hsv_histogram()
    print(
        "\nGenerated " + "\x1b[1;31m" + "{}".format(settings.model) + "\x1b[0m" +
        " histograms for all videos in directory '{}'".format(directory)
    )


def test_hist_classifier():
    """
    Prompts the user to crop the recorded video before generating an averaged BGR histogram and comparing it with the
    other averaged histograms for matching.
    :return: None
    """
    directory = "../recordings/"
    recordings = ["recording1.mp4", "recording2.mp4", "recording3.mp4"]  # 1: cloud-sky, 2: seal, 3: butterfly
    file = recordings[0]

    # calculate histogram for the recorded video
    print("\nPlease crop the recorded video for the histogram to be generated.")
    histogram_generator = HistogramGenerator(directory, file)
    if settings.model == "gray":
        histogram_generator.generate_video_grayscale_histogram(is_query=True)
    elif settings.model == "rgb":
        histogram_generator.generate_video_rgb_histogram(is_query=True)
    elif settings.model == "hsv":
        histogram_generator.generate_video_hsv_histogram(is_query=True)
    histogram_generator.match_histograms()
    print("Finished matching video using histogram comparison technique.")


def segment_video():
    directory = "../recordings/"
    video = "scene-segmentation.mp4"

    shot_boundary_detector = HistogramGenerator(directory, video)
    shot_boundary_detector.rgb_histogram_shot_boundary_detection()
    print("Finished detecting shot boundaries for {}.".format(video))


if __name__ == "__main__":
    main()
