
# Figuring out how to use ur-rtde by having robot end effector move in a straight line from a to b

import rtde_control
import rtde_receive
import cv2
from scipy.spatial.transform import Rotation as R

class controller():
    def __init__(self, start, end):
        # set up openCV window to use it's keyboard functions
        cv2.namedWindow("Controller", cv2.WINDOW_NORMAL)
        
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.56.101")




    def move(self, start, end, speed = 0.1 ):
        #move back and forth between start and end at speed, until user presses stop button on opencv window or or closes the window
        # rtde controls
        while True:
            self.rtde_c.moveL(start, speed) # move in a straight line to start
            self.rtde_c.moveL(end, speed)   # move in a straight line to end

            # check for user input to stop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC to quit
                print("Stopping controller...")
                break
            if key == ord("e"):  # 'e' to stop movement immediately
                print("Stopping movement...")
                self.rtde_c.stopL()
            if key == ord("s"): # 's' to start movement immediately
                print("Starting movement...")
                self.rtde_c.moveL(start, speed) # move in a straight line to start
                self.rtde_c.moveL(end, speed)   # move in a straight line to end


        self.rtde_c.stopL()  # stop any ongoing movement
        cv2.destroyAllWindows()

    def move_to_start(self, start, speed = 0.1):
        self.rtde_c.moveL(start, speed) # move in a straight line to start

def main():

    """
    cheat sheet.
    Bounds:
        X,    Y,   Z
    -0.75,0.90,0.20
    -0.30,0.90,0.00
    -0.75,-0.90,0.20
    -0.30,-0.90,0.20

    """
    pi = 3.14


    # Keeps the end effector pointing down while twisting it 45 degrees so the camera and Laser axis are parallel to the conveyor belt axis
        # Base: pointing down (rotate pi around X)
    r1 = R.from_rotvec([pi, 0, 0])

    # Twist: 45 degrees around Z
    r2 = R.from_euler('z', 45, degrees=True)

    # Combine rotations (order matters!)
    r_combined = r1 * r2

    # Convert back to rotation vector
    rotvec = r_combined.as_rotvec()

    #print(rotvec)


    start = [-0.50, 0, 0.30, rotvec[0], rotvec[1], rotvec[2]]
    end = [-0.50, 0.90, 0.30, rotvec[0], rotvec[1], rotvec[2]]

    ctrl = controller(start, end)
    ctrl.move(start, end)

    #ctrl.move_to_start(start)




if __name__ == "__main__":
    main()