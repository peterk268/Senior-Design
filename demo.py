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
import numpy as np
import tensorflow as tf

# Load the pre-trained model, here we are using the 640x640 set to increase accuracy.
model_dir = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_dir)

# In the model we are using, the tennis ball classification id is 37 
# which we will use to match to the found objects in the screen.
tennis_ball_class_id = 37

# Our match score here determines the threshold of a match to classify a tennis ball.
match_score = 0.35

# Function to perform object detection on each image in video frame.
def detect_objects(image):
    # We must resize the image to 640x640 to be compatible with the dataset.
    image = cv2.resize(image, (640, 640))
    image = image[np.newaxis, ...]

    input_tensor = tf.convert_to_tensor(image)
    detections = model(input_tensor)

    return detections

# Load video from the camera 
video = cv2.VideoCapture(0)

# Exit if video could not be opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Infinite Loop for video, analyzing each frame.
while True:
    # Read a new frame
    ok, image = video.read()
    if not ok:
        break

    # Perform object detection
    detections = detect_objects(image)

    # Process all the objects detected in the video.
    for i, score in enumerate(detections['detection_scores'][0].numpy()):
        # If we find an object that has a match score over 0.35 and it's id is equal to the tennis ball id.
        if score > match_score and int(detections['detection_classes'][0][i].numpy()) == tennis_ball_class_id:
            # We find the dimensions of the tennis ball detected and extract the frame around it.
            bbox = detections['detection_boxes'][0][i].numpy()
            ymin, xmin, ymax, xmax = bbox
            h, w, _ = image.shape
            xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

            # Here we draw a green rectangle overlaying the tennis ball that we detected.
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Tennis Ball Detection', image)

    # User can hit 'q' or ctrl + C to exit the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
