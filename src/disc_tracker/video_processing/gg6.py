import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from disc_tracker.video_processing import Tracker
from disc_tracker import DATA_DIRECTORY


def cleanMask(mask: np.ndarray) -> np.ndarray:
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask


# Blob detector parameters
params = cv.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = 16

# Read in video from file
video_chanel = "left"
video_resolution = {"x": 1280, "y": 720}
video = cv.VideoCapture(
    os.path.join(DATA_DIRECTORY, "rosie_pull", "video", f"{video_chanel}.mp4")
)
# Check video is open
if not video.isOpened():
    exit("Can't Open file")

background_subtractor = cv.createBackgroundSubtractorMOG2()  # Initialise BG subtractor
tracker = Tracker()  # Initialise tracker
blob_detector = cv.SimpleBlobDetector_create(params)  # Initialise blob detector

disc_id = 10


def main() -> None:
    while video.isOpened():
        ret, frame = video.read()  # Read next frame
        # Check if at end of file
        if not ret:
            print("End of file")
            break

        # Object detection
        foreground_mask = background_subtractor.apply(frame)  # Create FG mask for frame
        # Clean the mask to optimise object detection
        foreground_mask = cleanMask(foreground_mask)
        blobs = blob_detector.detect(foreground_mask)  # Blob detection
        if blobs == ():
            continue
        # Update the tracker with the (x,y) coords of each blob in the frame
        tracks = tracker.update(cv.KeyPoint_convert(blobs))
        # Plot blob locations and show IDs over the video frame
        frame = cv.drawKeypoints(
            image=frame,
            keypoints=blobs,
            outImage=np.array([]),
            color=(0, 0, 255),
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        for id in tracks:
            cv.putText(
                img=frame,
                text=f"{id}",
                # bottom left of text
                org=(tracks[id].position[0:2] - 10).astype(int),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
            )
        # Plot track history as line over the video frame
        if disc_id in tracks:
            disc_track = tracks[disc_id].track.astype(int)
            frame = cv.polylines(
                img=frame,
                pts=[np.array(disc_track[:, 0:2])],
                isClosed=False,
                color=(255, 0, 0),
                thickness=3,
                lineType=cv.LINE_AA,
            )

        cv.imshow(f"{video_chanel} camera", frame)
        if cv.waitKey(25) == ord("q"):
            break
    # Clean up
    video.release()
    cv.destroyAllWindows()

    # Plot the 2D track for the disc
    # Invert y to convert from (0,0) top-left to bottom-left
    x, y = disc_track[:, 0], abs(disc_track[:, 1] - video_resolution["y"])
    plt.plot(x, y)
    # Set axis to have dimensions of the video frame
    plt.axis("image")
    plt.xlim(0, video_resolution["x"])
    plt.ylim(0, video_resolution["y"])
    plt.show()

    # Save the track for the disc to file
    np.savez(
        os.path.join(DATA_DIRECTORY, "rosie_pull", "tracks", f"{video_chanel}.npz"),
        x=x,
        y=y,
        t=disc_track[:, 2],
    )


if __name__ == "__main__":
    main()
