import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

#Load pretained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "vid1.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        track_ids_list = []  # Initialize a list to store track IDs 

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs 
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks commit
        for box, track_id in zip(boxes, track_ids):

            if track_id not in track_ids_list:  #Ensure unique track IDs
                track_ids_list.append(track_id)#Adding track id to list
           
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
       
        #Display the annotated frame 
        cv2.namedWindow('YOLOv11 Tracking', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('YOLOv11 Tracking', annotated_frame)
        cv2.resizeWindow('YOLOv11 Tracking', 700, 500)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


