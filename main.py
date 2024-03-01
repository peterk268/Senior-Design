########################################
########    STORM               ########
########    Team #12            ########
########################################
########    Sahar Berlinerblaw  ########
########    Jimmy Fernandez     ########
########    Ewomazino Ebiala    ########
########    Peter Khouly        ########
########################################

import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time

def predict_trajectory_with_kalman(last_positions, num_future_steps, degree=None):
    if not last_positions or num_future_steps <= 0:
        raise ValueError("Invalid input for positions or number of future steps.")
    
    # Use a larger state space if more frames are provided
    n = len(last_positions)
    if n >= 4:  # Sufficient data for a constant acceleration model
        dt = 1  # Assuming time step is 1 for simplicity
        # State transition matrix for constant acceleration model
        F = np.array([[1, dt, 0.5*dt**2, 0,  0,         0],
                      [0,  1, dt,        0,  0,         0],
                      [0,  0, 1,         0,  0,         0],
                      [0,  0, 0,         1, dt, 0.5*dt**2],
                      [0,  0, 0,         0,  1, dt],
                      [0,  0, 0,         0,  0, 1]])
        
        H = np.array([[1, 0, 0, 0, 0, 0],  # Measurement matrix
                      [0, 0, 0, 1, 0, 0]])
        
        Q = np.eye(6) * 0.001  # Process noise covariance
        R = np.eye(2) * 10     # Measurement noise covariance
        P = np.eye(6) * 100    # Initial estimate error covariance
        
        # Initial state (position, velocity, acceleration for x and y)
        x = np.array([last_positions[0][0], 0, 0, last_positions[0][1], 0, 0])
    else:
        # Use a simpler model if not enough data is available for constant acceleration
        return predict_trajectory_with_kalman_simple_model(last_positions, num_future_steps, degree)

    # Use the polynomial fit as the measurement for Kalman update
    x_positions = [pos[0] for pos in last_positions]
    y_positions = [pos[1] for pos in last_positions]
    time_steps = np.arange(1, len(last_positions) + 1)
    
    degree = degree or min(len(last_positions) - 1, 3)  # Limit degree to 3 or less
    coefficients_x = np.polyfit(time_steps, x_positions, degree)
    coefficients_y = np.polyfit(time_steps, y_positions, degree)
    
    def calculate_position(coefficients, time):
        return sum(coefficient * (time ** power) for power, coefficient in enumerate(coefficients[::-1]))
    
    predictions = []
    for step in range(1, num_future_steps + 1):
        next_time = len(last_positions) + step
        z = np.array([calculate_position(coefficients_x, next_time),
                      calculate_position(coefficients_y, next_time)])
        x, P = kalman_predict_and_update(x, P, F, H, Q, R, z)
        predictions.append((x[0], x[3]))  # x[0] is position_x and x[3] is position_y
    
    return predictions

def kalman_predict_and_update(x, P, F, H, Q, R, z):
    # Prediction step
    x = F @ x
    P = F @ P @ F.T + Q
    
    # Update step
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    y = z - (H @ x)
    x = x + (K @ y)
    P = (np.eye(len(x)) - (K @ H)) @ P
    
    return x, P


# A simple linear motion model for trajectory prediction

def predict_trajectory_with_kalman_simple_model(last_positions, num_future_steps, degree=1):
    if not last_positions or num_future_steps <= 0:
        raise ValueError("Invalid input for positions or number of future steps.")
    
    # Initialize the Kalman filter components
    dt = 1  # Assuming time step is 1 for simplicity
    F = np.array([[1, dt, 0,  0],  # State transition matrix
                  [0,  1, 0,  0],  # for constant velocity model
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    
    H = np.array([[1, 0, 0, 0],  # Measurement matrix
                  [0, 0, 1, 0]])
    
    Q = np.eye(4) * 0.001  # Process noise covariance
    R = np.eye(2) * 10  # Measurement noise covariance
    P = np.eye(4) * 100  # Initial estimate error covariance
    
    # Initial state
    x = np.array([last_positions[0][0], 0, last_positions[0][1], 0])
    
    def kalman_predict_and_update(x, P, z):
        # Prediction step
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update step
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = z - (H @ x)
        x = x + (K @ y)
        P = (np.eye(len(x)) - (K @ H)) @ P
        
        return x, P
    
    # Using the polynomial fit as the measurement for Kalman update
    x_positions = [pos[0] for pos in last_positions]
    y_positions = [pos[1] for pos in last_positions]
    time_steps = np.arange(1, len(last_positions) + 1)
    
    degree = min(degree, len(last_positions) - 1)
    coefficients_x = np.polyfit(time_steps, x_positions, degree)
    coefficients_y = np.polyfit(time_steps, y_positions, degree)
    
    def calculate_position(coefficients, time):
        return sum(coefficient * (time ** power) for power, coefficient in enumerate(coefficients[::-1]))
    
    predictions = []
    for step in range(1, num_future_steps + 1):
        next_time = len(last_positions) + step
        z = np.array([calculate_position(coefficients_x, next_time),
                      calculate_position(coefficients_y, next_time)])
        x, P = kalman_predict_and_update(x, P, z)
        predictions.append((x[0], x[2]))  # x[0] is position_x and x[2] is position_y
    
    return predictions

# Tracker class to manage detected objects
class Tracker:
    def __init__(self, max_history=10, max_disappeared=20):
        self.next_object_id = 0
        self.objects = defaultdict(list)
        self.disappeared = defaultdict(int)
        self.max_history = max_history
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id].append(centroid)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return

        new_object_centroids = np.array([((r[0] + r[2]) // 2, (r[1] + r[3]) // 2) for r in rects])

        if len(self.objects) == 0:
            for i in range(0, len(new_object_centroids)):
                self.register(new_object_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id][-1] for object_id in object_ids]

            # Pairwise distance between new centroids and existing object centroids
            D = np.linalg.norm(np.array(object_centroids) - new_object_centroids[:, None], axis=2)

            # Find the smallest value in each row and then sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()

            # Similarly, find the smallest value in each column and then sort using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[col]
                self.objects[object_id].append(new_object_centroids[row])
                self.objects[object_id] = self.objects[object_id][-self.max_history:]  # maintain history size
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Deregister objects if they have disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(new_object_centroids[col])

    def get_predictions(self, num_future_steps):
        predictions = {}
        for object_id, centroids in self.objects.items():
            if len(centroids) > 1:
                predictions[object_id] = predict_trajectory_with_kalman(centroids, num_future_steps, degree=1)
        return predictions

# Initialize the YOLO model
model = YOLO('STORMx8.pt') #yolov8x STORMx8

# Initialize the Tracker
tracker = Tracker(max_history=5, max_disappeared=10)

# Initialize a dictionary to store the squares and timestamps
drawn_squares = {}

# Open the video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Couldn't open the video source.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read a frame from the video source.")
    exit()

# Initialize blank_image and colors here, after reading the first frame
last_predicted_points = {}  # Initialize last_predicted_points
overlay = np.zeros_like(frame)
square_timestamps = []  # Store timestamps of drawn squares

# Process the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Detect objects
    results = model.predict(source=frame, conf=0.09, show=False, classes=1, device='mps') #for classes and yolov8x use track
    rects = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for box in results[0].boxes]

    # Ensure rects are synchronized with tracker.objects
    if len(tracker.objects) != len(rects):
        # If the number of objects detected and tracked is different, reset the tracker
        tracker = Tracker(max_history=5, max_disappeared=10)

    tracker.update(rects)

    # Draw the detected boxes and add the centroid to the list

    for object_id, centroids in tracker.objects.items():
        if centroids:  # Ensure there is at least one centroid
            centroid = centroids[-1]
            text = "ID {}".format(object_id)
            org = (centroid[0] - 10, centroid[1] - 10)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Ensure the object_id is within the bounds of rects
            if 0 <= object_id < len(rects):
                last_rect = rects[object_id]
                cv2.rectangle(frame, (last_rect[0], last_rect[1]), (last_rect[2], last_rect[3]), (0, 255, 0), 2)

    # Iterate through detected objects and draw their boxes
    for object_id, centroids in tracker.objects.items():
        if centroids:  # Ensure there is at least one centroid
            centroid = centroids[-1]
            text = "ID {}".format(object_id)
            org = (centroid[0] - 10, centroid[1] - 10)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if 0 <= object_id < len(rects):
                last_rect = rects[object_id]
                cv2.rectangle(frame, (last_rect[0], last_rect[1]), (last_rect[2], last_rect[3]), (0, 255, 0), 2)

    # Predict future positions
    predictions = tracker.get_predictions(num_future_steps=10)
    for object_id, future_positions in predictions.items():
        if future_positions:
            # Draw circles for predicted future positions
            for (future_x, future_y) in future_positions:
                cv2.circle(frame, (int(future_x), int(future_y)), 10, (0, 0, 255), -2)
            # Update the last predicted point for this object
            last_predicted_points[object_id] = (future_positions[-1], time.time())

    # Go through all the points and draw those that are within the time window
    overlay_copy = overlay.copy()
    current_time = time.time()
    for object_id, (last_point, timestamp) in list(last_predicted_points.items()):
        if current_time - timestamp <= 2:
            (last_x, last_y) = last_point
            square_size = 50
            top_left = (int(last_x - square_size), int(last_y - square_size))
            bottom_right = (int(last_x + square_size), int(last_y + square_size))
                
            # Draw the rectangle on the frame
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                
            # Save drawn squares to dict
            drawn_squares[object_id] = {
                'top_left': top_left,
                'bottom_right': bottom_right,
                'color': (255, 0, 0),
                'thickness': 2,
                'timestamp': timestamp
                }
            square_timestamps.append(timestamp)  # Store timestamp in the list
            square_size = 25
            top_left = (int(last_x - square_size), int(last_y - square_size))
            bottom_right = (int(last_x + square_size), int(last_y + square_size))
            cv2.rectangle(overlay, top_left, bottom_right, (255, 50, 255), 2)

    # Erase squares that have exceeded the 2-second threshold
    for timestamp in square_timestamps.copy():  # Iterate over a copy of the list
        if current_time - timestamp > 2:
            index = square_timestamps.index(timestamp)
            if index < len(drawn_squares):  # Check if the index is within bounds
                object_id = list(drawn_squares.keys())[index]
                square_info = drawn_squares.pop(object_id)
                square_timestamps.pop(index)

                # Erase the old square by filling it with zeros in the overlay
                overlay.fill(0)
            else:
                # Handle the case where the index is out of bounds
                square_timestamps.pop(index)

    # Blend the overlay with the current frame
    frame_with_squares = cv2.addWeighted(frame, 1, overlay_copy, 1, 0)


    # Display the final frame with squares
    cv2.imshow("Object Tracking and Trajectory Prediction", frame_with_squares)

    # Clear the overlay after displaying the frame
    #overlay.fill(0)



    # Check for user input to close the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
