from ikpy.chain import Chain
import numpy as np

# Load SainSmart Arm's URDF model (replace with your actual URDF)
arm_chain = Chain.from_urdf_file("brazo.urdf")

# Target XYZ position
target_position = [0.1, 0.12, 0.1]  # (x, y, z) in meters

# Solve for joint angles
joint_angles = arm_chain.inverse_kinematics(target_position)

print("Joint Angles:", np.degrees(joint_angles))

#

import serial
import time

#arduino = serial.Serial('/dev/ttyUSB0', 9600)  # Adjust port if needed
time.sleep(2)  # Wait for connection

angles = target_position
angle_str = ",".join(map(str, angles)) + "\n"
#arduino.write(angle_str.encode())

print("Sent:", angle_str)
