import numpy as np
import torch
from flame import FlameDecoder
import pyrender
import trimesh
from config import get_config

from vtkplotter import Plotter, datadir, Text2D,show, interactive
import vtkplotter.mesh
import time

shape_params_size = 300
expression_params_size = 100

phong_shading = False # vtkplotter bug causes issues when using phong shading

def update_flame():
	t = time.time()
	vertice = flamelayer(shape_params, expression_params, pose_params, neck_pose, transl)
	time_took = time.time()-t
	print ('Time took = ', time_took)
	vertices = vertice[0].detach().cpu().numpy().squeeze()
	

	mesh.points(vertices) # This line can be used when there is no phong shading, otherwise vtkplotter crush (therefore we have the lines below, though they cause vtkplotter to "blink")
	"""
	global mesh
	vp.clear(mesh)
	# attempt to solve a vtkplotter bugs with phong shading
	
	mesh_n = vtkplotter.mesh.Mesh([vertices, faces]).computeNormals().phong()
	
	show(mesh_n, interactive=0)
	vp.clear(mesh)
	mesh = mesh_n
	interactive()
	"""
	
def flame_shape_slider(widget, event):
	value = widget.GetRepresentation().GetValue()
	global shape_params
	shape_params = torch.ones((config.batch_size,shape_params_size)).cuda()*value
	update_flame()

def flame_expr_slider(widget, event):
	value = widget.GetRepresentation().GetValue()
	global expression_params
	expression_params = torch.ones((config.batch_size,expression_params_size)).cuda()*value
	update_flame()

def flame_jaw_slider(widget, event):
	value = widget.GetRepresentation().GetValue()
	global pose_params
	pose_params[:,3:] = torch.ones((config.batch_size,pose_params[:,3:].shape[1])).cuda()*value
	update_flame()

def flame_neck_slider(widget, event):
	value = widget.GetRepresentation().GetValue()
	global neck_pose
	neck_pose = torch.ones((config.batch_size,3)).cuda()*value
	update_flame()

def switch_shading():
	global phong_shading
	phong_shading = not phong_shading

vp = Plotter(axes=0)

config = get_config()
config.batch_size = 1
config.flame_model_path = './model/male_model.pkl'
config.use_3D_translation = True # could be removed, depending on the camera model
config.use_face_contour = False


flamelayer = FlameDecoder(config)
flamelayer.cuda()
# Creating a batch of mean shapes
shape_params = torch.zeros((config.batch_size,shape_params_size)).cuda()

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotaation
# pose_params_numpy[:, 3:] : jaw rotaation
pose_params_numpy = np.array([[0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

# Cerating a batch of neutral expressions
expression_params = torch.zeros((config.batch_size,expression_params_size), dtype=torch.float32).cuda()
flamelayer.cuda()

print('Building Flame layer')
neck_pose = torch.zeros(config.batch_size,3).cuda()
#eye_pose = torch.zeros(config.batch_size,6).cuda() 
transl = torch.zeros(config.batch_size,3).cuda()
vertice = flamelayer(shape_params, expression_params, pose_params, neck_pose, transl)

faces = flamelayer.faces

vertices = vertice[0].detach().cpu().numpy().squeeze()

if phong_shading:
	mesh = vtkplotter.mesh.Mesh([vertices, faces]).computeNormals().phong()
else:
	mesh = vtkplotter.mesh.Mesh([vertices, faces]).flat()#computeNormals().phong()#flat()

vp.addSlider2D(flame_shape_slider, xmin=-1., xmax=1., value=0, pos=1, title="Flame shape")
vp.addSlider2D(flame_expr_slider, xmin=-1, xmax=1, value=0, pos=2, title="Flame expression")
vp.addSlider2D(flame_jaw_slider, xmin=-0.05, xmax=0.05, value=0, pos=3, title="Flame jaw")
vp.addSlider2D(flame_neck_slider, xmin=-0.05, xmax=0.05, value=0, pos=4, title="Flame neck")

vp += Text2D(__doc__)
vp.show(mesh)
