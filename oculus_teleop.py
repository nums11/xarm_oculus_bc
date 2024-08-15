import sys
import os
sys.path.append('/home/weirdlab/xarm_bc/xArm-Python-SDK')
from xarm.wrapper import XArmAPI
from time import sleep
from oculus_reader.oculus_reader.reader import OculusReader
import numpy as np

ip = '192.168.1.185'

arm = XArmAPI(port=ip)
arm.connect()
arm.clean_warn()
arm.clean_error()
arm.motion_enable(enable=True)
arm.set_mode(0) # position control mode
arm.set_state(state=0) # sport state
print("Resetting arm")
arm.reset(wait=True)
_, pos = arm.get_position()
arm.set_position(x=pos[0]+150, y=pos[1], z=pos[2],
    roll=pos[3], pitch=pos[4], yaw=pos[5], wait=True)
arm.set_mode(1) # servo mootion mode
arm.set_state(state=0) # sport state
sleep(2)

print("Initializing Oculus Reader")
oculus = OculusReader()
sleep(1)

def getArmPosition():
    code, [arm_x, arm_y, arm_z, roll, pitch, yaw] = arm.get_position()
    return np.array([arm_x, arm_y, arm_z, roll, pitch, yaw])

def getControllerPosition():
    transformations, _ = oculus.get_transformations_and_buttons()
    positions = transformations['r'][:,3]
    return positions[0:3]

orig_arm_pos = getArmPosition()
roll, pitch, yaw = orig_arm_pos[3:]
arm_pos = [orig_arm_pos[0], orig_arm_pos[1], orig_arm_pos[2]]
controller_pos = getControllerPosition()
print("Begin Teleoperation")
while True:
    new_controller_pos = getControllerPosition()

    controller_deltas = np.subtract(new_controller_pos,controller_pos) * 1000
    arm_deltas = [-1 * controller_deltas[2],
        controller_deltas[0], controller_deltas[1]]
    
    new_arm_pos = np.add(arm_pos, arm_deltas)
    arm.set_servo_cartesian(mvpose=[new_arm_pos[0],
        new_arm_pos[1], new_arm_pos[2], roll, pitch, yaw])
    
    controller_pos = new_controller_pos
    arm_pos = new_arm_pos
    sleep(0.005)