from controller import Robot, Motor, GPS, Gyro, Camera, InertialUnit
import numpy as np
import math
from typing import Dict, cast

# 1. Initialize Robot Context
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# 2. Set Up Motors
motors: Dict[str, Motor] = {
    'front_left': cast(Motor, robot.getDevice('front left propeller')),
    'front_right': cast(Motor, robot.getDevice('front right propeller')),
    'rear_left': cast(Motor, robot.getDevice('rear left propeller')),
    'rear_right': cast(Motor, robot.getDevice('rear right propeller'))
}

for motor in motors.values():
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

# 3. Set Up Sensors
gps = cast(GPS, robot.getDevice('gps'))
gps.enable(timestep)

gyro = cast(Gyro, robot.getDevice('gyro'))
gyro.enable(timestep)

imu = cast(InertialUnit, robot.getDevice('inertial unit'))
imu.enable(timestep)

camera = cast(Camera, robot.getDevice('camera'))
camera.enable(4 * timestep)

# --- PID FLIGHT TUNING CONSTANTS ---
HOVER_SPEED = 68.5      # Baseline motor RPM required to counteract gravity

K_P_ALT = 8.0           # Altitude proportional gain
K_D_ALT = 4.0           # Vertical velocity braking (stops vertical bouncing)
                        # ! Reduced K_P_ALT and K_D_ALT

K_P_POS = 0.2           # Horizontal positioning tracking speed
                        # ! Reduced K_P_POS
K_P_ATT = 16.0          # Leveling sharpness (Attitude correction force)
                        # ! Increased K_P_ATT
K_D_ANG = 4.0           # Rotational angular velocity dampening (Stops the wobbling)
K_YAW_D = 3.0           # Anti-spin angular brake (Stops the spinning)

MAX_ANGLE = 0.4         # Added MAX_ANGLE constant for conveniance

past_z = 0.12
first_frame = True

'''
def get_target_position():
    """Reads coordinates injected into CustomData by the supervisor node."""
    custom_data = robot.getCustomData()
    if not custom_data:
        return None
    try:
        x, y, z = map(float, custom_data.split(','))
        return np.array([x, y, z])
    except ValueError:
        return None
'''

def get_target_velocity():
    custom_data = robot.getCustomData()
    if not custom_data:
        return None
    try:
        dx, dy, dz = map(float, custom_data.split(','))
        return np.array([dx, dy, dz])
    except ValueError:
        return None

# --- MAIN CONTROL LOOP ---
while robot.step(timestep) != -1:


    '''
    # Read Sensors
    pos = np.array(gps.getValues())
    ang_vel = np.array(gyro.getValues())    # [roll_rate, pitch_rate, yaw_rate]
    rpy = np.array(imu.getRollPitchYaw())   # [Roll, Pitch, Yaw]

    # Guard Clause against Webots NaN Initialization Trap
    if np.isnan(pos).any() or np.isnan(ang_vel).any() or np.isnan(rpy).any():
        continue

    # FIX: Account for the Mavic 2 Pro's default internal IMU roll frame offset
    roll = rpy[0] # ! Removed " + pi/2"
    pitch = rpy[1]

    if first_frame:
        past_z = pos[2]
        first_frame = False
        
    # Track vertical velocity for the D-term damping
    dt = timestep / 1000.0
    vertical_vel = (pos[2] - past_z) / dt
    past_z = pos[2]
    
    # Resolve Path Target
    target = get_target_position()
    if target is None:
        target = np.array([0.0, 0.0, 3.0])  # Fallback vector if supervisor isn't ready
        
    error = target - pos  # [Error_X, Error_Y, Error_Z]
    
    # --- CASCADING CONTROL LOGIC ---
    
    # 1. Altitude Control (Thrust)
    thrust = HOVER_SPEED + (K_P_ALT * error[2]) - (K_D_ALT * vertical_vel)
    
    # 2. Position Control -> Mapping XY position errors into target frame angles
    # ! This needs to angle away before you reach the target position. Currently, it always angles towards the target position, meaning it speeds up until it passes the position. It needs to slow down before it reaches the position.
    yaw = rpy[2]

    c = math.cos(yaw)
    s = math.sin(yaw)

    body_x = c * error[0] + s * error[1]
    body_y = -s * error[0] + c * error[1]

    target_pitch = np.clip(+K_P_POS * body_x, -MAX_ANGLE, MAX_ANGLE) # ! Made this positive instead of negative
    target_roll  = np.clip(-K_P_POS * body_y, -MAX_ANGLE, MAX_ANGLE) # ! Adjusted target pitch and roll for yaw; Made this negative instead of positive
    
    # 3. Attitude & Gyro Dampening Control (Stops the violent wobbling)
    roll_input = K_P_ATT * (roll - target_roll) + K_D_ANG * ang_vel[0]
    pitch_input = K_P_ATT * (target_pitch - pitch) - K_D_ANG * ang_vel[1] # ! Switched the target_pitch and pitch
    
    # 4. Anti-Spin Correction (Stops the spinning)
    yaw_input = K_YAW_D * ang_vel[2] # ! Made this positive instead of negative
    
    # 5. Fixed Mixer Matrix for Webots Mavic 2 Pro Structure
    fl = thrust - roll_input - pitch_input + yaw_input
    fr = thrust + roll_input - pitch_input - yaw_input
    rl = thrust - roll_input + pitch_input - yaw_input
    rr = thrust + roll_input + pitch_input + yaw_input
    
    # 6. Apply Motor Limits & Handle Reverse Thrust Properties for Counter-Rotators
    motors['front_left'].setVelocity(np.clip(fl, 0.0, 95.0))
    motors['front_right'].setVelocity(np.clip(-fr, -95.0, 0.0))
    motors['rear_left'].setVelocity(np.clip(-rl, -95.0, 0.0))
    motors['rear_right'].setVelocity(np.clip(rr, 0.0, 95.0))
    '''