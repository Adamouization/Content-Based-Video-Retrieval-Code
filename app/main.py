from app.helpers import get_video_filenames
from app.histogram import HistogramGenerator


def main():
    """
    Program entry point.
    :return: None
    """
    train_hist_classifier()
    test_hist_classifier()


def train_hist_classifier():
    """
    Generates an averaged BGR histogram for all the videos in the directory-based database.
    :return: None
    """
    directory = "../footage/"
    files = get_video_filenames(directory)

    for file in files:
        print("generating histogram for {}".format(file))
        histogram_generator = HistogramGenerator(directory, file)
        histogram_generator.generate_video_histograms()
    print("Generated histograms for all files in directory {}".format(directory))


def test_hist_classifier():
    """
    Prompts the user to crop the recorded video before generating an averaged BGR histogram and comparing it with the
    other averaged histograms for matching.
    :return: None
    """
    directory = "../recordings/"
    file = "recording.mp4"

    # calculate histogram for the recorded video
    print("Please crop the recorded video for the histogram to be generated.")
    histogram_generator = HistogramGenerator(directory, file)
    histogram_generator.generate_recording_video_histograms()
    histogram_generator.match_histograms()
    print("Finished matching video using histogram comparison technique.")


if __name__ == "__main__":
    main()
