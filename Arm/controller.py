import torch
import cv2
import numpy as np
import time
from ikpy.chain import Chain
from ikpy.link import URDFLink
import Jetson.GPIO as GPIO

# Load the YOLO model
model = torch.load('yolo11x_raspberry_model.pt')
model.eval()

# Initialize stereo cameras
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

# Camera calibration parameters (replace with your own calibration data)
focal_length = 700  # Example focal length
baseline = 0.06  # Distance between cameras in meters

# Set up stereo matcher
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Define GPIO pins for the robotic arm servos
servo_pins = [18, 23, 24, 25, 12, 16]  # Example GPIO pins for 6DOF (update as needed)
GPIO.setmode(GPIO.BOARD)
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)

# Initialize the robotic arm chain (6DOF)
arm_chain = Chain(name='arm', links=[
    URDFLink(name="base", translation_vector=[0, 0, 0.1], rotation=[0, 0, 0]),
    URDFLink(name="shoulder", translation_vector=[0, 0, 0.1], rotation=[0, 1, 0]),
    URDFLink(name="elbow", translation_vector=[0.1, 0, 0], rotation=[0, 1, 0]),
    URDFLink(name="wrist1", translation_vector=[0.1, 0, 0], rotation=[0, 1, 0]),
    URDFLink(name="wrist2", translation_vector=[0.1, 0, 0], rotation=[0, 1, 0]),
    URDFLink(name="end_effector", translation_vector=[0.1, 0, 0], rotation=[0, 0, 0]),
])

# Function to calculate depth from disparity
def calculate_depth(disparity):
    depth = (focal_length * baseline) / disparity
    return depth

# Function to move the robotic arm to a target position
def move_arm_to_position(target_position):
    joint_angles = arm_chain.inverse_kinematics(target_position)
    for i, angle in enumerate(joint_angles[1:]):  # Skip base index [0]
        duty_cycle = 2.5 + (np.degrees(angle) / 180) * 10  # Convert angle to duty cycle
        pwm[i].ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)

# Function to open the gripper
def open_gripper():
    # Replace with actual command to open gripper
    GPIO.output(gripper_pin, GPIO.HIGH)

# Function to close the gripper
def close_gripper():
    # Replace with actual command to close gripper
    GPIO.output(gripper_pin, GPIO.LOW)

# Fixed position to place raspberries
placement_position = [0.5, 0.0, 0.0]  # Adjust as needed

# Initialize PWM for servos
pwm = [GPIO.PWM(pin, 50) for pin in servo_pins]
for p in pwm:
    p.start(0)  # Start with 0 duty cycle

# Assuming you have a gripper connected to a specific pin
gripper_pin = 21  # Example GPIO pin for gripper
GPIO.setup(gripper_pin, GPIO.OUT)

while True:
    # Capture frames
    retL, frameL = left_camera.read()
    retR, frameR = right_camera.read()
    if not retL or not retR:
        break

    # Preprocess frames for YOLO detection
    input_frameL = cv2.resize(frameL, (640, 480))

    # Run YOLO detection on the left image
    results = model(input_frameL)
    boxes = results.pandas().xyxy[0]  # Get detected boxes

    # Disparity map calculation
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Process each detected raspberry
    for index, box in boxes.iterrows():
        x1, y1, x2, y2, conf, cls = box[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
        if conf > 0.5:  # Confidence threshold
            bbox_disparity = disparity[int(y1):int(y2), int(x1):int(x2)]
            avg_disparity = np.mean(bbox_disparity[bbox_disparity > 0])  # Avoid zero disparity
            if avg_disparity > 0:
                depth = calculate_depth(avg_disparity)
                raspberry_position = [x1, y1, depth]  # Adjust based on your coordinate system
                print(f"Detected raspberry at depth: {depth:.2f} meters")

                # Move to the raspberry position
                move_arm_to_position(raspberry_position)

                # Close the gripper to grab the raspberry
                close_gripper()
                time.sleep(1)  # Adjust duration for the grabbing action

                # Move to the fixed placement position
                move_arm_to_position(placement_position)
                time.sleep(1)  # Adjust duration for placing action

                # Open the gripper to release the raspberry
                open_gripper()
                time.sleep(1)  # Adjust duration for the releasing action

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for p in pwm:
    p.stop()
GPIO.cleanup()
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()
