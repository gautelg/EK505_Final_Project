import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("ISS_env.xml")
data = mujoco.MjData(model)
viewer.launch(model, data)
