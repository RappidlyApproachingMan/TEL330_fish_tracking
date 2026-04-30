import rtde_receive
import time

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")

print("Reading TCP pose every second. Move robot manually or just observe.")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        tcp = rtde_r.getActualTCPPose()
        print(f"X={tcp[0]:.4f}  Y={tcp[1]:.4f}  Z={tcp[2]:.4f}  "
              f"Rx={tcp[3]:.4f}  Ry={tcp[4]:.4f}  Rz={tcp[5]:.4f}")
        time.sleep(1.0)

except KeyboardInterrupt:
    print("Done.")
    rtde_r.disconnect()