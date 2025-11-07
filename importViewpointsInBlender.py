import bpy
import json

# Path to JSON file exported from Python
json_file = "C:/full/path/to/viewpoints_tsp.json"

# Load viewpoints
with open(json_file, "r") as f:
    viewpoints = json.load(f)

# Optional: Create a new collection to organize spheres
if "Viewpoints" not in bpy.data.collections:
    vp_collection = bpy.data.collections.new("Viewpoints")
    bpy.context.scene.collection.children.link(vp_collection)
else:
    vp_collection = bpy.data.collections["Viewpoints"]

# Create spheres or empty objects at each viewpoint
for i, vp in enumerate(viewpoints):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=vp)
    sphere = bpy.context.object
    sphere.name = f"Viewpoint_{i}"
    # Move sphere into collection
    vp_collection.objects.link(sphere)
    bpy.context.scene.collection.objects.unlink(sphere)
    
    # Optional: color the sphere
    mat = bpy.data.materials.new(name=f"VP_Mat_{i}")
    mat.diffuse_color = (0.2, 0.8, 0.2, 1)
    sphere.data.materials.append(mat)
