import numpy as np
import cv2 as cv
from tracker import *

tracker = EuclideanDistTracker()
# Blob detector parameters
params = cv.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False
params.minArea = 10

video = cv.VideoCapture('.\DiscTracker\Video\RosiePull R.mp4')

fgbg = cv.createBackgroundSubtractorMOG2()

if not video.isOpened():
    print("Can't Open file")

tracks = []
while video.isOpened():
    ret, frame = video.read()
    fgMask = fgbg.apply(frame)
    if not ret:
        print('End of file')
        break
    cv.imshow('Frame', frame)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25))
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)

    blobDetector = cv.SimpleBlobDetector_create(params)

    blobs = blobDetector.detect(fgMask)
    tracks.append(cv.KeyPoint_convert(blobs))

    ids = tracker.update(tracks)
    print(ids)
    fgMask_with_blobs = cv.drawKeypoints(fgMask, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow('FG Mask', fgMask_with_blobs)
    if cv.waitKey(25) == ord('q'):
        break

video.release()
cv.destroyAllWindows()