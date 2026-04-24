import rtde_receive
import time


rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.101")
while True:
    print(rtde_r.getActualTCPPose())
    time.sleep(1)

    # when "s" is pressed take a snapshot of the current pose and save it to a file
    key = input("Press 's' to save current pose, 'p' to pass to next pose or 'q' to quit: ")
    if key == "s":
        pose = rtde_r.getActualTCPPose()
        with open("tcp_bounds.txt", "a") as f:
            f.write(",".join(map(str, pose)) + "\n")
        print("Pose saved:", pose)
    elif key == "p":
        print("Passing to next pose...")
    elif key == "q":
        print("Exiting...")
        break