import bpy
import os


currPath = os.path.splitext(bpy.data.filepath)[0]+ ".curves.js"
file = open(currPath, "w") 

file.write('var curves = {\n')
for ob in bpy.data.objects.values() : 
  if ob.type == 'CURVE' :
    file.write( '"%s":\n' % ob.name)
    for spline in ob.data.splines :
      if len(spline.bezier_points) > 0 :
        file.write("[")
        for bezier_point in spline.bezier_points.values() : 
          handle_left  = ob.matrix_world * bezier_point.handle_left
          co           = ob.matrix_world * bezier_point.co
          handle_right = ob.matrix_world * bezier_point.handle_right

          file.write("[[%.3f, %.3f, %.3f],  " % (handle_left.x, handle_left.y, handle_left.z ))
          file.write("[%.3f, %.3f, %.3f],  " % (co.x, co.y, co.z ))
          file.write("[%.3f, %.3f, %.3f]],\n  " % (handle_right.x, handle_right.y, handle_right.z ))

    file.write("],\n")
file.write("}\n")
file.close()