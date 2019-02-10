import os


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
