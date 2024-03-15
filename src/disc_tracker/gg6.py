import os

import numpy as np
import cv2 as cv
from SimpleTracker import *
import matplotlib.pyplot as plt

def cleanMask(mask):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

# Blob detector parameters
params = cv.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False
params.minArea = 10
# Read in video from file
data_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data")
video = cv.VideoCapture(os.path.join(data_directory, "rosie_pull", "video", "right.mp4"))
# Check video is open
if not video.isOpened():
    print("Can't Open file")

# out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (1280,720))
fgbg = cv.createBackgroundSubtractorMOG2()  # Initialise BG subtractor
tracker = SimpleDistanceTracker()  # Initialise tracker
blobDetector = cv.SimpleBlobDetector_create(params)  # Initialise blob detector

while video.isOpened():
    ret, frame = video.read()  # Read next frame
    # Check if at end of file
    if not ret:
        print('End of file')
        break
    fgMask = fgbg.apply(frame)  # Create FG mask for frame
    fgMask = cleanMask(fgMask)  # Clean the mask to optimise object detection
    # cv.imshow('Frame', frame)
    blobs = blobDetector.detect(fgMask)  # Blob detection
    tracks = tracker.update(cv.KeyPoint_convert(blobs))  # Update the tracker with the (x,y) coords of each blob in the frame
    # Plot blob locations and show IDs over the FG mask image
    fgMask_with_blobs = cv.drawKeypoints(fgMask, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(len(tracks.keys())):
        try:
            text = f"{list(tracks.keys())[i]}"  # IDs
            loc = tracks[i][-1]  # Most recent location of the blob (current end of track)
            cv.putText(frame, text, (int(loc[0]-10), int(loc[1]-10)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        except:
            pass
    # cv.imshow('FG Mask', fgMask_with_blobs)
    # Plot track history as line over the video frame
    try:
        discIndex = list(tracks.keys()).index(13)
        discTrack = tracks[discIndex]
        discTrack = [list(discTrack[i]) for i in range(len(discTrack))]
        frame = cv.polylines(frame, [np.array(discTrack, np.int32).reshape(-1, 1, 2)], False, (255, 0, 0), 3, cv.LINE_AA)
        cv.imshow('Frame', frame)
    except:
        cv.imshow('Frame', frame)
    # out.write(frame)
    if cv.waitKey(25) == ord('q'):
        break
# Clean up
video.release()
# out.release()
cv.destroyAllWindows()

# Plot the 2D track for the disc
discIndex = list(tracks.keys()).index(13)
discTrack = tracks[discIndex]
x, y = zip(*discTrack)
y = [abs(oldy-720) for oldy in y]  # Invert y to convert from (0,0) top-left to bottom-left
plt.plot(x, y)
# Set axis to have dimensions of the video frame
plt.axis('image')
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.show()
# Save the track for the disc to file
np.savez(os.path.join(data_directory, "rosie_pull", "tracks", "right.npz"), x=x, y=y)