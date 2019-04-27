import cv2
from pyspin.spin import make_spin, Box1
from vidstab import VidStab


class ClickAndDrop:
    """
    Class for selecting a region of interest on a frame by click and dropping the mouse over the desired area, and
    cropping that frame to include the pixels inside the ROI only.
    """
    frame_size = (1280, 720)
    window_name = "Crop the recording (top-left click -> bottom-right drop) - 'C' to crop - 'R' to restart"

    def __init__(self, thumbnail):
        """
        Loads the image to crop and controls the callback loop. Calculates the reference points once a valid cropped
        area has been chosen and the user has clicked "c".
        :param thumbnail: the first frame of the video to crop
        """
        self.thumbnail = thumbnail

        self.reference_points = list()
        self.cropping = False

        # load the image, clone it, and setup the mouse callback function
        self.image = cv2.resize(self.thumbnail, self.frame_size, interpolation=cv2.INTER_AREA)
        clone = self.image.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):  # reset the cropping region
                self.image = clone.copy()
            elif key == ord("c"):  # break from the loop
                break

        # if there are 2 reference points, then crop the region of interest from the image
        if len(self.reference_points) == 2:
            self.roi = clone[self.reference_points[0][1]:self.reference_points[1][1],
                             self.reference_points[0][0]:self.reference_points[1][0]]

        # close all open windows
        cv2.destroyAllWindows()

    def click_and_crop(self, event, x, y, flags, param):
        """
        Callback function allowing a user to manually crop an image.

        NOTE: must crop from top-left corner to bottom-right corner

        :param event: one of the MouseEventTypes constants
        :param x: the x-coordinate of the mouse event
        :param y: the y-coordinate of the mouse event
        :param flags: one of the MouseEventFlags constants
        :param param: optional parameters
        :return: None
        """
        # if the left mouse button was clicked, record the starting (x, y) coordinates and indicate that cropping is
        # being performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.reference_points = [(x, y)]
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that the cropping operation is finished
            self.reference_points.append((x, y))
            self.cropping = False

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.reference_points[0], self.reference_points[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

    def get_roi(self):
        """
        Return the selected region of interest of the frame.
        :return: roi: the user-cropped region of the image
        """
        return self.roi

    def get_reference_points(self):
        """
        Returns the ROI's reference points coordinates
        :return: reference_points: a list containing the two x and y coordinates used to crop the image.
        """
        return self.reference_points


class VideoStabiliser:
    """
    Class used to stabilise the recorded video for more optimal matching.
    """
    def __init__(self, directory, file_name):
        """
        Initialise variables and call the function to stabilise the specified video.
        :param directory: the directory where the video file to stabilise is located
        :param file_name: the mp4 video file's name
        """
        self.directory = directory
        self.file = file_name
        self.new_file = file_name[:-4]

        self.stabiliser = VidStab()
        self.stabilise_video()

    @make_spin(Box1, "Stabilising video...")
    def stabilise_video(self):
        """
        Stabilises a mp4 video and outputs the result as an avi file in the same directory.
        :return:
        """
        self.stabiliser.stabilize(input_path="{}{}".format(self.directory, self.file),
                                  output_path="{}/stable-{}.avi".format(self.directory, self.new_file),
                                  border_type="reflect")
        print("\nVideo stabilised!")
