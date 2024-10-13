import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# Constants for Eye Aspect Ratio (EAR) calculation
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

# Frame rate target (15 fps)
FPS = 15
FRAME_DELAY = int(1000 / FPS)  # Time per frame in milliseconds

# Load the pre-trained models for face detection and landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate the head pose (to check distraction)
def get_head_pose(shape):
    image_points = np.array([
        shape[33],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left Mouth corner
        shape[54]   # Right Mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-165.0, 170.0, -135.0),     # Left eye left corner
        (165.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right Mouth corner
    ])

    focal_length = 1 * frame.shape[1]
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    
    return p1, p2

# Start video stream from webcam
cap = cv2.VideoCapture(0)

# Store the last frame processing time
last_frame_time = 0

while True:
    # Get the current time
    current_time = time.time()

    # If sufficient time has passed for 15 fps, process the frame
    if (current_time - last_frame_time) >= 1.0 / FPS:
        last_frame_time = current_time
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray, 0)
        
        for face in faces:
            # Get the facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Extract coordinates of left and right eye
            leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
            
            # Compute EAR for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            # Average the EAR values for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
            # Draw the eyes on the frame
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Head pose detection
            p1, p2 = get_head_pose(shape)
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Check if EAR is below the threshold, implying eyes are closed (sleepy)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                
                # If eyes were closed for sufficient frames, alert the driver
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "SLEEPY! Take a break!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
            
            # Check distraction by analyzing head pose
            if p2[1] > p1[1] + 20:  # Assuming downward tilt or away from road
                cv2.putText(frame, "DISTRACTED! Focus on the road!", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Driver Monitoring System", frame)

    # Add a delay to keep frame rate at max 15 fps
    if cv2.waitKey(FRAME_DELAY) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
