# simplify_watertight_scaled.py — Blender 4.5 headless-safe
import bpy, os, sys

# ---- CONFIG ----
INPUT  = os.path.abspath("ISS_Stationary_Bare.glb")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "cube_project", "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_SIMPLIFIED = os.path.join(DATA_DIR, "iss_wt_simplified.obj")
OUTPUT_RAW        = os.path.join(DATA_DIR, "iss_raw_export.obj")

VOXEL_SIZE     = 0.5
SCALE_FACTOR   = 1.0
SMOOTH_ITERS   = 10
SMOOTH_LAMBDA  = 0.5
DECIMATE_RATIO = 0.10
WELD_THRESH    = 1.0

def set_active(obj):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)

if not os.path.isfile(INPUT):
    print(f"[ERROR] Missing input: {INPUT}")
    sys.exit(1)

# Import GLB
bpy.ops.import_scene.gltf(filepath=INPUT)
print("[OK] Imported:", INPUT)

# Ensure object mode
if bpy.ops.object.mode_set.poll():
    bpy.ops.object.mode_set(mode='OBJECT')

# Collect meshes
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    print("[ERROR] No mesh objects found after import.")
    sys.exit(1)

# 1) Join into a single object first
set_active(meshes[0])
for o in meshes[1:]:
    o.select_set(True)
bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# 2) Apply existing transforms
set_active(obj)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# 3) Scale the object
obj.scale = (SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR)

# 4) Apply the scale so geometry is truly scaled before remesh
set_active(obj)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# ----------------------------------------------------------------------
# NEW STEP: duplicate the pre-simplified mesh and export it to OBJ
# ----------------------------------------------------------------------
raw_copy = obj.copy()
raw_copy.data = obj.data.copy()
bpy.context.scene.collection.objects.link(raw_copy)

# Export raw copy only
set_active(raw_copy)
bpy.ops.wm.obj_export(
    filepath=OUTPUT_RAW,
    export_selected_objects=True,   # only the raw_copy
    export_eval_mode='DAG_EVAL_VIEWPORT',
    export_triangulated_mesh=True
)
print("[OK] Raw mesh exported:", OUTPUT_RAW)

# Remove the raw copy so it does NOT end up inside the simplified OBJ
bpy.data.objects.remove(raw_copy, do_unlink=True)

# Switch back to main object
set_active(obj)
# ----------------------------------------------------------------------

# --- 5) Voxel Remesh (watertight) ---
rem = obj.modifiers.new("Remesh", 'REMESH')
rem.mode = 'VOXEL'
rem.voxel_size = VOXEL_SIZE
set_active(obj)
bpy.ops.object.modifier_apply(modifier=rem.name)

# --- 6) Laplacian Smooth ---
lap = obj.modifiers.new("LapSmooth", 'LAPLACIANSMOOTH')
lap.iterations = SMOOTH_ITERS
lap.lambda_factor = SMOOTH_LAMBDA
set_active(obj)
bpy.ops.object.modifier_apply(modifier=lap.name)

# --- 7) Decimate ---
dec = obj.modifiers.new("Decimate", 'DECIMATE')
dec.ratio = DECIMATE_RATIO
set_active(obj)
bpy.ops.object.modifier_apply(modifier=dec.name)

# --- 8) Weld ---
weld = obj.modifiers.new("Weld", 'WELD')
weld.merge_threshold = WELD_THRESH
set_active(obj)
bpy.ops.object.modifier_apply(modifier=weld.name)

# --- 8.1) triangulate ---
tri = obj.modifiers.new("Triangulate", 'TRIANGULATE')
set_active(obj)
bpy.ops.object.modifier_apply(modifier=tri.name)

# Smooth shading
bpy.ops.object.shade_smooth()

# Export simplified OBJ (only remaining mesh object in scene)
set_active(obj)
bpy.ops.wm.obj_export(
    filepath=OUTPUT_SIMPLIFIED,
    export_selected_objects=True,   # only this object
    export_eval_mode='DAG_EVAL_VIEWPORT',
    export_triangulated_mesh=True
)
print("[OK] Simplified mesh exported:", OUTPUT_SIMPLIFIED)
