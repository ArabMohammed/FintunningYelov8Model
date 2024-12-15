from ultralytics import YOLO
import numpy as np
import cv2
from pytube import YouTube
import yt_dlp
##################################################
def download_youtube_video(youtube_url, output_path='video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Force MP4
        'outtmpl': output_path,  # File name of the output
        'merge_output_format': 'mp4'  # Final format will be .mp4
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
###############################################################
###############################################################
video_file_name='youtube_video.mp4'
youtube_url = 'https://youtu.be/4dvqnW7vLE4?si=JUI5kdp36X8WUPr7'  # Change this URL to your desired video
# Step 1: Download the video
video_path = download_youtube_video(youtube_url,video_file_name)
# Define the threshold for considering traffic as heavy
###########################################
video_file_name='sample_video.mp4'
#############################
best_model = YOLO('Model01.pt') 
#############################
heavy_traffic_threshold = 10
# Define the vertices for the quadrilaterals
# points coordinates for points that constitutes observation areas on right and left
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for the slice and lane threshold
x1, x2 = 325, 635 
lane_threshold = 609

# Define the positions for the text annotations on the image
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

METERS_PER_PIXEL = 0.05
# Open the video
cap = cv2.VideoCapture(video_file_name)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#########
out = cv2.VideoWriter('traffic_density_analysis.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
########
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
# Read until video is completed
compter = 0
while cap.isOpened() and compter < 100:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Create a copy of the original frame to modify
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 
        detection_frame = frame.copy()
    
        # Black out the regions outside the specified vertical range
        detection_frame[:x1, :] = 0  # Black out from top to x1
        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)
        
        # Restore the original top and bottom parts of the frame
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()        
        
        # Draw the quadrilaterals on the processed frame
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Retrieve the bounding boxes from the results
        bounding_boxes = results[0].boxes

        # Initialize counters for vehicles in each lane
        vehicles_in_left_lane = 0
        vehicles_in_right_lane = 0
        ################################
        current_bounding_boxes = []
        # Loop through each bounding box to count vehicles in each lane
        for box in bounding_boxes.xyxy:
            # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
            x_min = box[0]  
            y_min = box[1]  
            x_max = box[2]  
            y_max = box[3]  
            current_bounding_boxes.append((x_min, y_min, x_max, y_max))

            #######################################
            if box[0] < lane_threshold:
                vehicles_in_left_lane += 1
            else:
                vehicles_in_right_lane += 1
        #############################################
        if frame_number > 1:
            for i, curr_box in enumerate(current_bounding_boxes):
                if i < len(previous_bounding_boxes):
                    prev_box = previous_bounding_boxes[i]
                    y1 = (curr_box[1] + curr_box[3]) / 2  # Current center y position
                    y2 = (prev_box[1] + prev_box[3]) / 2  # Previous center y position
                    distance_pixels = abs(y1 - y2)  # Calculate pixel displacement
                    
                    # Convert pixel displacement to meters
                    distance_meters = distance_pixels * METERS_PER_PIXEL

                    # Calculate speed in meters per second (m/s)
                    speed_mps = distance_meters * fps
                    
                    # Convert speed from m/s to km/h
                    speed_kmh = speed_mps * 3.6
                    
                    # Display the speed on the frame
                    cv2.putText(processed_frame, f'Speed: {speed_kmh:.2f} km/h', 
                                (int(curr_box[0]), int(curr_box[1] - 10)), 
                                font, 0.5, font_color, 1, cv2.LINE_AA)

        previous_bounding_boxes = current_bounding_boxes             
        out.write(processed_frame)
        # Uncomment the following 3 lines if running this code on a local machine to view the real-time processing results
        # cv2.imshow('Real-time Analysis', processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q on keyboard to exit the loop
        #     break
        compter+=1
    else:
        break

# Release the video capture and video write objects
cap.release()
out.release()
