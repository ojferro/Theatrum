import mujoco
import mujoco.viewer

# Don't use scientific notation
import numpy as np
np.set_printoptions(suppress=True)

m = mujoco.MjModel.from_xml_path('./cart_pole_simplified.xml')
d = mujoco.MjData(m)

viewer = mujoco.viewer.launch_passive(m, d)
mujoco.mj_step(m, d)
viewer.sync()


id = mujoco.mj_name2id(m, 1, "beam")
print(f"ID: {id}")
print(f"Mass: {m.body_mass[id]}")
print(f"Inertias: {m.body_inertia}")
print(f"Body Inertia: {m.body_inertia[id]}")
print(f"CoM Positions: {d.xipos}")
print(f"Body CoM Position: {d.xipos[id]}")

dirX, dirY, dirZ = 0, 1, 2

# Calculate the inertia about the wheel axle using the CoM position. Use parallel axis theorem:
# I = I_cm + m * r^2

# CoM position
r = d.xipos[id][dirZ]
M = m.body_mass[id]
I_com = m.body_inertia[id][dirY]

I_axle = I_com + M * r**2

print(f"Inertia about axle: {I_axle}")

# Solve for equivalent point mass system
# I = M* L**2
L = np.sqrt(I_axle/M)

print(f"Equivalent point mass system has mass: {M} and L: {L}")

# THis is not working.... It for some reason reports inertia in the body? geom? something frame...
