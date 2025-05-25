import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time
from collections import deque
import os

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    a, b, c: lists or arrays of x,y coordinates
    Returns angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class JointAngleTracker:
    def __init__(self, video_path=0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video source: {video_path}")
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.angle_history = {
            'right_knee': deque(maxlen=5),
            'right_elbow': deque(maxlen=5)
        }
        self.rep_count = {
            'squat': 0,
            'pushup': 0
        }
        self.stage = {
            'squat': None,
            'pushup': None
        }
        self.csv_file = open('joint_angles.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Right Knee Angle', 'Right Elbow Angle'])
    
    def smooth_angle(self, angle_list):
        if len(angle_list) == 0:
            return 0
        return sum(angle_list) / len(angle_list)
    
    def get_landmark_coords(self, landmarks, idx):
        return [landmarks[idx].x, landmarks[idx].y]
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Unable to read frame from video source")
            return False, None
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            if not results.pose_landmarks:
                return True, image  # Skip frame if no landmarks detected
            landmarks = results.pose_landmarks.landmark

            # Calculate bounding box of landmarks
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            bbox_center_x = (min_x + max_x) / 2
            bbox_center_y = (min_y + max_y) / 2

            # Define central ROI (center 50% width and height)
            roi_min_x, roi_max_x = 0.25, 0.75
            roi_min_y, roi_max_y = 0.25, 0.75

            # Check if bbox center is within central ROI
            if not (roi_min_x <= bbox_center_x <= roi_max_x and roi_min_y <= bbox_center_y <= roi_max_y):
                return True, image  # Skip frame if person not in center

            # Right knee angle
            hip = self.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
            knee = self.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
            ankle = self.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            knee_angle = calculate_angle(hip, knee, ankle)
            self.angle_history['right_knee'].append(knee_angle)
            smooth_knee_angle = self.smooth_angle(self.angle_history['right_knee'])
            
            # Right elbow angle
            shoulder = self.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            elbow = self.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            wrist = self.get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            self.angle_history['right_elbow'].append(elbow_angle)
            smooth_elbow_angle = self.smooth_angle(self.angle_history['right_elbow'])
            
            # Repetition counting logic for squat
            if smooth_knee_angle > 140:
                self.stage['squat'] = "up"
            if smooth_knee_angle < 90 and self.stage['squat'] == "up":
                self.stage['squat'] = "down"
                self.rep_count['squat'] += 1
            
            # Repetition counting logic for pushup
            if smooth_elbow_angle > 160:
                self.stage['pushup'] = "up"
            if smooth_elbow_angle < 90 and self.stage['pushup'] == "up":
                self.stage['pushup'] = "down"
                self.rep_count['pushup'] += 1
            
            # Display angles and reps
            h, w, _ = image.shape
            
            # Knee angle display
            knee_cx = int(knee[0] * w)
            knee_cy = int(knee[1] * h)
            cv2.putText(image, f'Knee Angle: {int(smooth_knee_angle)}', (knee_cx, knee_cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Elbow angle display
            elbow_cx = int(elbow[0] * w)
            elbow_cy = int(elbow[1] * h)
            cv2.putText(image, f'Elbow Angle: {int(smooth_elbow_angle)}', (elbow_cx, elbow_cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Feedback messages
            if smooth_knee_angle < 90:
                squat_status = "Low Squat"
            elif 90 <= smooth_knee_angle < 140:
                squat_status = "Good Squat Form"
            else:
                squat_status = "Standing"
            
            if smooth_elbow_angle < 90:
                pushup_status = "Down Position"
            elif 90 <= smooth_elbow_angle < 160:
                pushup_status = "Mid Position"
            else:
                pushup_status = "Up Position"
            
            cv2.putText(image, f'Squat: {squat_status} Reps: {self.rep_count["squat"]}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f'Pushup: {pushup_status} Reps: {self.rep_count["pushup"]}', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Write data to CSV
            timestamp = time.time()
            self.csv_writer.writerow([timestamp, int(smooth_knee_angle), int(smooth_elbow_angle)])
            
            cv2.imshow('Joint Angle Tracker', image)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return True, image
    
    def run(self):
        while self.cap.isOpened():
            success, _ = self.process_frame()
            if not success:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = os.path.expanduser("~\\Downloads\\WhatsApp Video 2025-05-14 at 8.51.01 PM.mp4")
    tracker = JointAngleTracker(video_path)
    tracker.run()
