# simplify_watertight_scaled.py  — Blender 4.5 headless-safe
import bpy, os, sys

# ---- CONFIG ----
INPUT  = os.path.abspath("ISS_stationary_bare.glb")
OUTPUT = os.path.abspath("iss_wt_simplified.glb")

VOXEL_SIZE     = 0.5      # meters
SCALE_FACTOR   = 2.5       # 1.0 = no scaling
SMOOTH_ITERS   = 8
SMOOTH_LAMBDA  = 0.12
DECIMATE_RATIO = 0.3
WELD_THRESH    = 0.2      # meters
# ---------------

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

# Optional: triangulate
tri = obj.modifiers.new("Triangulate", 'TRIANGULATE')
set_active(obj)
bpy.ops.object.modifier_apply(modifier=tri.name)

# Smooth shading
bpy.ops.object.shade_smooth()

# Export GLB
bpy.ops.export_scene.gltf(
    filepath=OUTPUT,
    export_format='GLB',
    use_selection=False,
    export_apply=True
)
print("[OK] Exported:", OUTPUT)
