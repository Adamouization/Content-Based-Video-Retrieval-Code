import cv2


video_capture = cv2.VideoCapture('../animations/output/circle_blue_right.avi')

if not video_capture.isOpened():
    print("Error opening video file")

# read video until completion or user exit
while video_capture.isOpened():

    # read capture frame by frame
    ret, frame = video_capture.read()

    if ret:
        # display current frame
        cv2.imshow('Frame', frame)
        # user exit on "q" key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# tidying upq
video_capture.release()
cv2.destroyAllWindows()
