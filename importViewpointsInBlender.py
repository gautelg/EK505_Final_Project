import bpy
import json
import mathutils

# ===============================
# 1️⃣ LOAD SPACE STATION OBJ MODEL
# ===============================
obj_path = r"E:\EK505_introToRobotics\finalProject\EK505_Final_Project\Station_Model\iss_wt_simplified.obj"

# Remove all existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
bpy.ops.import_scene.obj(filepath=obj_path)
station = bpy.context.selected_objects[0]
print(f"[INFO] Imported OBJ: {station.name}")

# ===============================
# 2️⃣ LOAD VIEWPOINTS JSON
# ===============================
json_path = r"E:\EK505_introToRobotics\finalProject\EK505_Final_Project\optimized_viewpoints.json"
with open(json_path, "r") as f:
    data = json.load(f)

viewpoints = data["viewpoints"]
tsp_path = data["tsp_path"]

# ===============================
# 3️⃣ ADD SPHERES AT VIEWPOINTS
# ===============================
for i, vp in enumerate(viewpoints):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=vp)
    sphere = bpy.context.active_object
    sphere.name = f"VP_{i}"
    # Optional: assign green material
    mat = bpy.data.materials.new(name=f"VP_Mat_{i}")
    mat.diffuse_color = (0.2, 0.8, 0.2, 1.0)
    sphere.data.materials.append(mat)

# ===============================
# 4️⃣ ADD CAMERA
# ===============================
bpy.ops.object.camera_add()
camera = bpy.context.active_object
camera.name = "TSP_Camera"

# Set render resolution
scene = bpy.context.scene
scene.camera = camera
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = len(tsp_path) * 10  # 10 frames per segment

# ===============================
# 5️⃣ ANIMATE CAMERA ALONG TSP PATH
# ===============================
def set_camera_look_at(cam_obj, target):
    """Make camera look at target"""
    direction = mathutils.Vector(target) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

frames_per_segment = 10
frame_num = 1

for idx in tsp_path:
    vp = viewpoints[idx]
    camera.location = vp
    # Look at center of model
    center = station.location
    set_camera_look_at(camera, center)
    camera.keyframe_insert(data_path="location", frame=frame_num)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_num)
    frame_num += frames_per_segment

# ===============================
# 6️⃣ OPTIONAL: RENDER SETTINGS
# ===============================
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.filepath = r"E:\EK505_introToRobotics\finalProject\EK505_Final_Project\tsp_animation.mp4"

# ===============================
# 7️⃣ RENDER ANIMATION
# ===============================
bpy.ops.render.render(animation=True)
print("[INFO] Render complete!")
