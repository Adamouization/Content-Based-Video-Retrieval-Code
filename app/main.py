import os

from app.histogram import generate_video_histogram


def main():
    directory = "../footage/"
    for file in get_video_filenames(directory):
        print("generating histogram for {}".format(file))
        generate_video_histogram(directory, file)
    print("Generated histograms for all files in directory {}".format(directory))


def get_video_filenames(directory):
    list_of_videos = list()
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            list_of_videos.append(filename)
            # f = os.path.join(directory, filename)
        else:
            print("no mp4 files found in directory '{}'".format(directory))
    return list_of_videos


if __name__ == "__main__":
    main()
