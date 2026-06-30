from controller import Supervisor
import math

NUM_DRONES = 2  # Change this to 2, 4, 9, etc. It won't overlap anymore!
SPACING = 2.5
TARGET_HEIGHT = 3.0

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

drones = []

for i in range(NUM_DRONES):
    if i == 0:
        drone = supervisor.getFromDef('drone_0')
        drones.append(drone)
        continue
    
    # Clean linear spacing along the X axis on the ground (Z = 0.12)
    x = i * SPACING
    y = 0.0
    z = 0.12
    
    drone_string = f'Mavic2Pro {{ name "drone_{i}" translation {x} {y} {z} controller "drone_controller" }}'
    
    root = supervisor.getRoot()
    children_field = root.getField('children')
    children_field.importMFNodeFromString(-1, drone_string)
    
    drone = supervisor.getFromDef(f'drone_{i}')
    drones.append(drone)

start_time = supervisor.getTime()

while supervisor.step(timestep) != -1:
    current_time = supervisor.getTime() - start_time
    for i, drone in enumerate(drones):
        if drone is None:
            continue
        
        # Orbit paths mapped cleanly to the X-Y ground plane
        angle = (2 * math.pi * i / len(drones)) + (current_time * 0.3)
        radius = 4.0
        target_x = radius * math.cos(angle)
        target_y = radius * math.sin(angle)
        target_z = TARGET_HEIGHT + math.sin(current_time + i) * 0.3
        
        pos_field = drone.getField('customData')
        if pos_field:
            pos_field.setSFString(f"{target_x},{target_y},{target_z}")