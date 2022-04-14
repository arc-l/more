import bpy

content = ""
for a in bpy.context.selected_objects:
    content += '' + a.name + " "
    content += str(a.location[0])+' '+str(a.location[1])+' '+str(a.location[2]) + " "
    content += str(a.rotation_euler[0])+' '+str(a.rotation_euler[1])+' '+str(a.rotation_euler[2])
    content += "\n"

with open("/home/mluser/search-YCB/test-cases/ycb/Output.txt", "w") as text_file:
    text_file.write(content)