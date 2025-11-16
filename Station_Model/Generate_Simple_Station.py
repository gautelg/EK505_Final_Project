import bpy
from pathlib import Path

# Parameters for the cylinder
length = 10.0
radius = 2.0
faces = 32

output_path = Path(r"c:\Users\gpiga\Desktop\EK505_Final_Project\Station_Model\simple_station.obj")

# Clean scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create the cylinder
bpy.ops.mesh.primitive_cylinder_add(
    vertices=faces,
    radius=radius,
    depth=length,
    enter_editmode=False,
    align='WORLD',
    location=(0.0, 0.0, 0.0)
)

cylinder = bpy.context.object

# Ensure cylinder is the only selected and active object
bpy.ops.object.select_all(action='DESELECT')
cylinder.select_set(True)
bpy.context.view_layer.objects.active = cylinder

# --- TRIANGULATE THE MESH ---
# Switch to Edit Mode
bpy.ops.object.mode_set(mode='EDIT')

# Select all faces
bpy.ops.mesh.select_all(action='SELECT')

# Triangulate (quad/ngon -> triangles)
bpy.ops.mesh.quads_convert_to_tris(
    quad_method='BEAUTY',
    ngon_method='BEAUTY'
)

# Back to Object Mode
bpy.ops.object.mode_set(mode='OBJECT')
# --------------------------------

# Export OBJ (Blender 4.x operator)
bpy.ops.wm.obj_export(
    filepath=str(output_path),
    export_selected_objects=True,
    export_normals=True,
    export_uv=True,
    export_materials=False,
    path_mode='AUTO'
)

print(f"Cylinder exported to {output_path}")
