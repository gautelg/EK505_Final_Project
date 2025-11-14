import bpy
import json
import mathutils
import os

# ===============================
# 1) LOAD SPACE STATION OBJ MODEL
# ===============================

# Change these to your actual absolute paths or use Blender-relative paths.
# Example: obj_path = bpy.path.abspath("//Station_Model/iss_wt_simplified.obj")
obj_path = r"Station_Model\iss_wt_simplified.obj"

# Make sure the path exists to avoid silent failures
if not os.path.isfile(obj_path):
    raise FileNotFoundError(f"OBJ not found at: {obj_path}")

# Remove all existing objects in the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ - Blender 4.x uses wm.obj_import, not import_scene.obj
bpy.ops.wm.obj_import(filepath=obj_path)

# After import, the imported objects should be selected
imported_objs = list(bpy.context.selected_objects)
if not imported_objs:
    raise RuntimeError("OBJ import succeeded, but no objects are selected. Check the importer and filepath.")

# Use the first imported object as the station "center"
station = imported_objs[0]
print(f"[INFO] Imported OBJ: {station.name}")

# ===============================
# 2) LOAD VIEWPOINTS JSON
# ===============================
json_path = r"optimized_viewpoints.json"
if not os.path.isfile(json_path):
    raise FileNotFoundError(f"JSON not found at: {json_path}")

with open(json_path, "r") as f:
    data = json.load(f)

viewpoints = data["viewpoints"]
tsp_path = data["tsp_path"]

# Sanity checks
if not viewpoints:
    raise ValueError("No viewpoints found in JSON.")
if not tsp_path:
    raise ValueError("TSP path is empty in JSON.")

# ===============================
# 3) ADD SPHERES AT VIEWPOINTS
# ===============================
for i, vp in enumerate(viewpoints):
    # vp is expected to be [x, y, z]
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=vp,
        align='WORLD'  # explicit align for 4.x
    )
    sphere = bpy.context.active_object
    sphere.name = f"VP_{i}"

    # Create a simple material and assign it
    mat = bpy.data.materials.new(name=f"VP_Mat_{i}")
    # In 4.x, Material.diffuse_color is still a 4-float RGBA array
    mat.diffuse_color = (0.2, 0.8, 0.2, 1.0)
    if sphere.data.materials:
        sphere.data.materials[0] = mat
    else:
        sphere.data.materials.append(mat)

# ===============================
# 8) EXPORT COMBINED OBJ
# ===============================
# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Set the export path
export_path = os.path.join(os.path.dirname(obj_path), "station_with_viewpoints.obj")

# Export to OBJ
bpy.ops.wm.obj_export(
    filepath=export_path,
    export_selected_objects=True,
    forward_axis='Y',
    up_axis='Z'
)

print(f"[INFO] Exported combined model to: {export_path}")

# # ===============================
# # 4) ADD CAMERA
# # ===============================
# # Remove any existing cameras just to be clean
# for obj in list(bpy.data.objects):
#     if obj.type == 'CAMERA':
#         bpy.data.objects.remove(obj, do_unlink=True)

# bpy.ops.object.camera_add(align='WORLD', location=(0, 0, 0))
# camera = bpy.context.active_object
# camera.name = "TSP_Camera"

# # Set render resolution and basic scene settings
# scene = bpy.context.scene
# scene.camera = camera
# scene.render.resolution_x = 1920
# scene.render.resolution_y = 1080
# scene.render.fps = 24

# frames_per_segment = 10
# scene.frame_start = 1
# scene.frame_end = len(tsp_path) * frames_per_segment

# # ===============================
# # 5) ANIMATE CAMERA ALONG TSP PATH
# # ===============================
# def set_camera_look_at(cam_obj, target):
#     """Make camera look at a 3D point `target`."""
#     # target can be a Vector or a 3-tuple
#     target_vec = mathutils.Vector(target)
#     direction = target_vec - cam_obj.location
#     rot_quat = direction.to_track_quat('-Z', 'Y')
#     cam_obj.rotation_euler = rot_quat.to_euler()

# frame_num = 1

# # Use the station's origin as the look-at target
# center = station.location.copy()

# for idx in tsp_path:
#     # idx is an index into the viewpoints list
#     vp = viewpoints[idx]

#     # Set camera position
#     camera.location = vp
#     # Look at the station center
#     set_camera_look_at(camera, center)

#     # Insert keyframes
#     camera.keyframe_insert(data_path="location", frame=frame_num)
#     camera.keyframe_insert(data_path="rotation_euler", frame=frame_num)

#     frame_num += frames_per_segment

# # ===============================
# # 6) RENDER SETTINGS
# # ===============================
# scene.render.image_settings.file_format = 'FFMPEG'
# scene.render.ffmpeg.format = 'MPEG4'
# # You can also specify codec explicitly if you like:
# # scene.render.ffmpeg.codec = 'H264'

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scene.render.filepath = os.path.join(SCRIPT_DIR, "tsp_animation.mp4")

# # ===============================
# # 7) RENDER ANIMATION
# # ===============================
# bpy.ops.render.render(animation=True)
# print("[INFO] Render complete!")
