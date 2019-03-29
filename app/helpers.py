import os

from terminaltables import DoubleTable


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


def print_finished_training_message(model, directory, runtime):
    """
    Prints a message at the end of the training function.
    :param model: the histogram model used for training
    :param directory: the directory where the training dataset is
    :param runtime: the time elapsed in seconds
    :return: None
    """
    print(
        "\n\nGenerated " + "\x1b[1;31m" + "{}".format(model) + "\x1b[0m" +
        " histograms for all videos in directory '{}'".format(directory)
    )
    print("\n--- Runtime: {} seconds ---".format(runtime))

