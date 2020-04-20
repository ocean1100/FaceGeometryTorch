import os
import cv2
import sys
import argparse
import numpy as np
import trimesh
from vtkplotter import *
import vtkplotter.mesh

show_texture = True
def show_hide_texture():
    global show_texture
    show_texture = not show_texture
    mesh = load_mesh_obj_and_texture(args.input_folder, meshes[0])
    show(mesh)

def animate_meshes():
    mesh = load_mesh_obj_and_texture(args.input_folder, meshes[0])
    show(mesh, interactive=0)
    for mesh_p in meshes:
        mesh = load_mesh_obj_and_texture(args.input_folder, mesh_p)
        show(mesh)
    interactive()

def load_mesh_obj_and_texture(input_folder, mesh):
    obj_path = os.path.join(input_folder,mesh)
    text_path = os.path.splitext(obj_path)[0] + '.png'
    global show_texture
    if (show_texture):
        return vp.load(obj_path).clean().computeNormals().phong().texture(text_path)    
    else:
        return vp.load(obj_path).clean().computeNormals().phong()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize face video reconstruction')
    parser.add_argument('--input_folder', default='VOCA_results', help='Path of the input folder images')
    
    parser.add_argument('--mesh_viewpoint_ending',default='26_C.obj', help='Ending of the file from the given angle')

    args = parser.parse_args()

    # Get all images
    meshes = [mesh for mesh in os.listdir(args.input_folder) if mesh.endswith(args.mesh_viewpoint_ending)]
    meshes.sort()
    
    # uncomment the following to create a movie from the images
    #save_images_in_video(images, args.input_folder, args.output_folder, args.image_viewpoint_ending)

    vp = Plotter(axes=0)
    
    bu = vp.addButton(
    animate_meshes,
    pos=(0.7, 0.05),  # x,y fraction from bottom left corner
    states=["press to animate"],
    c=["w"],
    bc=["dg", "dv"],  # colors of states
    font="courier",   # arial, courier, times
    size=25,
    bold=True,
    italic=False,
    )

    texture_button = vp.addButton(
    show_hide_texture,
    pos=(0.3, 0.05),  # x,y fraction from bottom left corner
    states=["show texture", "hide texture"],
    c=["w"],
    bc=["dg", "dv"],  # colors of states
    font="courier",   # arial, courier, times
    size=25,
    bold=True,
    italic=False,
    )

    mesh = load_mesh_obj_and_texture(args.input_folder, meshes[0])
    vp.show(mesh)