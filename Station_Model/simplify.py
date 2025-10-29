# simplify_watertight_final.py  — Blender 4.5 headless-safe
import bpy, os, sys
import re

# ---- CONFIG ----
INPUT  = os.path.abspath("ISS_stationary_bare.glb")
OUTPUT = os.path.abspath("iss_wt_simplified.glb")
# Smallest feature size you want to preserve (meters). Lower -> more detail.
VOXEL_SIZE     = 0.35
# Voxel “stair-step” smoothing; harmless if small.
SMOOTH_ITERS   = 8
SMOOTH_LAMBDA  = 0.12
# Face reduction after remesh (0.20 = keep 20% of faces)
DECIMATE_RATIO = 0.1
# Merge extremely close verts (meters)
WELD_THRESH    = 0.05
# ---------------

def active(obj):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj

# Reset to an empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)

if not os.path.isfile(INPUT):
    print(f"[ERROR] Missing input: {INPUT}")
    sys.exit(1)

# Import glTF/GLB
bpy.ops.import_scene.gltf(filepath=INPUT)
print("[OK] Imported:", INPUT)

# Ensure object mode
if bpy.ops.object.mode_set.poll():
    bpy.ops.object.mode_set(mode='OBJECT')

# Gather mesh objects
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    print("[ERROR] No mesh objects found after import.")
    sys.exit(1)

# Apply transforms so VOXEL_SIZE is interpreted in meters
for o in meshes:
    active(o)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Join into a single object so the remesh produces one watertight envelope
active(meshes[0])
for o in meshes[1:]:
    o.select_set(True)
bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# --- 1) Voxel Remesh (watertight) ---
rem = obj.modifiers.new("Remesh", 'REMESH')
rem.mode = 'VOXEL'
rem.voxel_size = VOXEL_SIZE
# (4.5 has consistent props; avoid optional/removed ones)
active(obj)
bpy.ops.object.modifier_apply(modifier=rem.name)

# --- 2) Laplacian Smooth (reduce voxel stepping) ---
lap = obj.modifiers.new("LapSmooth", 'LAPLACIANSMOOTH')
lap.iterations = SMOOTH_ITERS
lap.lambda_factor = SMOOTH_LAMBDA
active(obj)
bpy.ops.object.modifier_apply(modifier=lap.name)

# --- 3) Decimate (face reduction) ---
dec = obj.modifiers.new("Decimate", 'DECIMATE')
dec.ratio = DECIMATE_RATIO
active(obj)
bpy.ops.object.modifier_apply(modifier=dec.name)

# --- 4) Weld (merge near-duplicate verts) ---
weld = obj.modifiers.new("Weld", 'WELD')
weld.merge_threshold = WELD_THRESH
active(obj)
bpy.ops.object.modifier_apply(modifier=weld.name)

# Optional: triangulate for planner robustness (keeps engines happy)
tri = obj.modifiers.new("Triangulate", 'TRIANGULATE')
active(obj)
bpy.ops.object.modifier_apply(modifier=tri.name)

# Smooth shading (no deprecated flags in 4.5)
bpy.ops.object.shade_smooth()

# --- Export GLB ---
bpy.ops.export_scene.gltf(
    filepath=OUTPUT,
    export_format='GLB',
    use_selection=False,
    export_apply=True
)
print("[OK] Exported:", OUTPUT)
