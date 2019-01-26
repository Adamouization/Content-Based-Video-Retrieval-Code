import cv2

video_capture = cv2.VideoCapture('../footage/1-waves.mp4')

frame_size = (1280, 720)

if not video_capture.isOpened():
    print("Error opening video file")

# print the number of frames in the video
frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
print(frame_count)

# read video until completion or user exit
while video_capture.isOpened():

    # read capture frame by frame
    ret, frame = video_capture.read()

    if ret:
        # resize and display current frame
        resized_frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        cv2.imshow('Frame', resized_frame)

        # user exit on "q" or "Esc" key press
        k = cv2.waitKey(30) & 0xFF
        if k == 25 or k == 27:
            break
    else:
        break

# tidying up
video_capture.release()
cv2.destroyAllWindows()
