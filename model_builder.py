import time

import mujoco
import mujoco.viewer


m = mujoco.MjModel.from_xml_path('./cart_pole_simplified.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

        # time.sleep(0.0001)
        time.sleep(1000)