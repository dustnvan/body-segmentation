import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Holistic model
    results = holistic.process(rgb_frame)

    # Draw the segmentation on the frame
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
        body_segment_labels = {
            (0, 0): "Nose",
            (4, 8): "Eyes",
            (1, 3): "Eyes",
            (9, 10): "Mouth",
            (11, 12): "Chest",
            (11, 13): "Bicep",
            (12, 14): "Bicep",
            (13, 15): "Forearm",
            (14, 16): "Forearm",
            (18, 20): "Palms",
            (15, 17): "Palms",
            (12, 23): "Torso",
            (24, 28): "Legs",
            (23, 27): "Legs"


        }
        # Add labels to each limb
        for connection, label in body_segment_labels.items():
            start_point, end_point = connection
        
        # Testing for plotting points
        # for connection in mp_holistic.POSE_CONNECTIONS:
        #     start_point = connection[0]
        #     end_point = connection[1]

            start_landmark = results.pose_landmarks.landmark[start_point]
            end_landmark = results.pose_landmarks.landmark[end_point]

            h, w, _ = frame.shape
            start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
            end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)

            # Calculate the midpoint for labeling
            mid_x, mid_y = (start_x + end_x) // 2, (start_y + end_y) // 2

            # Shows points Testing
            # cv2.putText(frame, f"{start_point}-{end_point}", (mid_x, mid_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(frame, label, (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Real-time Body Segmentation", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
