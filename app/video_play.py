import cv2


# video_capture = cv2.VideoCapture('../recordings/recording_circle_red_right.mov')
video_capture = cv2.VideoCapture('../animations/output/circle_red_left.avi')

if not video_capture.isOpened():
    print("Error opening video file")

# read video until completion or user exit
while video_capture.isOpened():

    # read capture frame by frame
    ret, frame = video_capture.read()

    if ret:
        # display current frame
        cv2.imshow('Frame', frame)

        # user exit on "q" or "Esc" key press
        k = cv2.waitKey(30) & 0xFF
        if k == 25 or k == 27:
            break
    else:
        break

# tidying up
video_capture.release()
cv2.destroyAllWindows()
