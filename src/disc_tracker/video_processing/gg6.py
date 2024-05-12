import os
from typing import OrderedDict

import cv2 as cv
import numpy as np

from disc_tracker.video_processing import Tracker


def cleanMask(mask: np.ndarray) -> np.ndarray:

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask


def setup_blob_detector() -> cv.SimpleBlobDetector:

    # Blob detector parameters
    params = cv.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 16

    return cv.SimpleBlobDetector_create(params)


def load_video(directory: str, chanel: str) -> cv.VideoCapture:

    filepath = os.path.join(directory, "video", f"{chanel}.mp4")
    video = cv.VideoCapture(filepath)
    # Check video is open
    if not video.isOpened():
        exit(f"Can't open file '{filepath}'")

    return video


def detect_objects(
    background_subtractor: cv.BackgroundSubtractorMOG2,
    blob_detector: cv.SimpleBlobDetector,
    frame: cv.typing.MatLike,
):

    foreground_mask = background_subtractor.apply(frame)  # Create FG mask for frame
    # Clean the mask to optimise object detection
    foreground_mask = cleanMask(foreground_mask)

    return blob_detector.detect(foreground_mask)  # Blob detection


def add_object_ids_to_frame(frame: cv.typing.MatLike, tracks: dict) -> None:

    for id, track in tracks.items():
        cv.putText(
            img=frame,
            text=f"{id}",
            # bottom left of text
            org=(track.position[0:2] - 10).astype(int),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
        )


def add_object_bbox_to_frame(frame: cv.typing.MatLike, blobs) -> cv.typing.MatLike:

    # Plot blob locations and show IDs over the video frame
    return cv.drawKeypoints(
        image=frame,
        keypoints=blobs,
        outImage=np.array([]),
        color=(0, 0, 255),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def track_objects(video: cv.VideoCapture, chanel: str) -> OrderedDict:

    background_subtractor = (
        cv.createBackgroundSubtractorMOG2()
    )  # Initialise BG subtractor
    blob_detector = setup_blob_detector()  # Initialise blob detector
    tracker = Tracker()  # Initialise tracker
    while video.isOpened():
        ret, frame = video.read()  # Read next frame
        # Check if at end of file
        if not ret:
            print("End of file")
            break

        blobs = detect_objects(background_subtractor, blob_detector, frame)
        if blobs == ():
            continue
        # Update the tracker with the (x,y) coords of each blob in the frame
        tracks = tracker.update(cv.KeyPoint_convert(blobs))

        frame = add_object_bbox_to_frame(frame, blobs)
        add_object_ids_to_frame(frame, tracks)

        cv.imshow(f"{chanel} camera", frame)
        if cv.waitKey(25) == ord("q"):
            break

    # Clean up
    video.release()
    cv.destroyAllWindows()

    return tracks


def save_disc_track(filename, tracks: dict, id: int) -> None:
    # Save the track for the disc to file
    np.savez(
        filename,
        x=tracks[id].track[:, 0],
        y=tracks[id].track[:, 1],
        t=tracks[id].track[:, 2],
    )
