import os
import cv2
import sys
import argparse
import numpy as np
import trimesh
from vtkplotter import *
import vtkplotter.mesh
import re

show_texture = True
def show_hide_texture():
    global show_texture
    show_texture = not show_texture
    mesh = load_mesh_obj_and_texture(args.input_folder, meshes[0])
    if show_texture:
            mesh.lighting('default', ambient, diffuse, specular)#, specularPower, specularColor)
    show(mesh)

def animate_meshes():
    ambient, diffuse, specular = 1., 1., 1.
    mesh = load_mesh_obj_and_texture(args.input_folder, meshes[0])
    mesh.lighting('default', ambient, diffuse, specular)#, specularPower, specularColor)
    show(mesh, interactive=0)
    for mesh_p in meshes:
        mesh = load_mesh_obj_and_texture(args.input_folder, mesh_p)
        if show_texture:
            mesh.lighting('default', ambient, diffuse, specular)#, specularPower, specularColor)
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


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize face video reconstruction')
    parser.add_argument('--input_folder', default='VOCA_results', help='Path of the input folder images')

    args = parser.parse_args()

    # Get all images
    meshes = [mesh for mesh in os.listdir(args.input_folder) if mesh.endswith('.obj')]
    # sort the list (such that 10 will be after 9 and not after 1.. as it is by string sorting)
    meshes = natural_sort(meshes)

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
    ambient, diffuse, specular = 1., 1., 1.
    mesh.lighting('default', ambient, diffuse, specular)#, specularPower, specularColor)
    vp.show(mesh)