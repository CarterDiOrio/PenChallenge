from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
# measured position

base_to_gripper = np.matrix([
    [1,  0,  0,  0.25177093],
    [0,   1, 0, -0.00078599],
    [0,  0.0,    1,  0.16086889],
    [0.0,          0.0,          0.0,          1.0]])
camera_to_gripper = np.linalg.inv(np.matrix([
    [1, 0, 0, 280.0 / 1000.0],
    [0, 1, 0, 540.0 / 1000.0],
    [0, 0, 1, 0.0 / 1000.0],
    [0, 0, 0, 1]
]))

# robot = InterbotixManipulatorXS("px100", "arm", "gripper")
# print(robot.arm.get_ee_pose())

# print(camera_to_gripper)


def point_to_base(point):
    return base_to_gripper * camera_to_gripper * point


# The robot object is what you use to control the robot
# mode = 'h'
# # Let the user select the position
# while mode != 'q':
#     mode = input("[h]ome, [s]leep, [q]uit ")
#     if mode == "h":
#         robot.arm.go_to_home_pose()
#         print(robot.arm.get_ee_pose())
#     elif mode == "s":
#         robot.arm.go_to_sleep_pose()
#         print(robot.arm.get_ee_pose())
