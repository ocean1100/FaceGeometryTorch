import re
import os
import sys
import numpy as np

def write_obj(mesh, filename, flip_faces=False, group=False, comments=None):
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    ff = -1 if flip_faces else 1

    def write_face_to_obj_file(face_index, obj_file):
        vertex_indices = mesh.f[face_index][::ff] + 1

        if hasattr(mesh, 'ft'):
            texture_indices = mesh.ft[face_index][::ff] + 1
            if not hasattr(mesh, 'fn'):
                mesh.reset_face_normals()
            normal_indices = mesh.fn[face_index][::ff] + 1
            #obj_file.write('f %d/%d/%d %d/%d/%d  %d/%d/%d\n' % tuple(
            #    np.array([vertex_indices, texture_indices, normal_indices]).T.flatten()))
            obj_file.write('f %d/%d %d/%d  %d/%d\n' % tuple(
                np.array([vertex_indices, texture_indices]).T.flatten()))
        elif hasattr(mesh, 'fn'):
            normal_indices = mesh.fn[face_index][::ff] + 1
            obj_file.write('f %d//%d %d//%d  %d//%d\n' % tuple(np.array([vertex_indices, normal_indices]).T.flatten()))
            print ('there')
        else:
            obj_file.write('f %d %d %d\n' % tuple(vertex_indices))

    with open(filename, 'w') as fi:
        if comments is not None:
            if isinstance(comments, str):
                comments = [comments]
            for comment in comments:
                for line in comment.split("\n"):
                    fi.write("# %s\n" % line)

        if hasattr(mesh, 'texture_filepath'):
            outfolder = os.path.dirname(filename)
            outbase = os.path.splitext(os.path.basename(filename))[0]
            mtlpath = outbase + '.mtl'
            fi.write('mtllib %s\n' % mtlpath)
            from shutil import copyfile
            texture_name = outbase + os.path.splitext(mesh.texture_filepath)[1]
            if os.path.abspath(mesh.texture_filepath) != os.path.abspath(os.path.join(outfolder, texture_name)):
                copyfile(mesh.texture_filepath, os.path.join(outfolder, texture_name))
            mesh.write_mtl(os.path.join(outfolder, mtlpath), outbase, texture_name)

        for r in mesh.v:
            fi.write('v %f %f %f\n' % (r[0], r[1], r[2]))

        if hasattr(mesh, 'fn') and hasattr(mesh, 'vn'):
            for r in mesh.vn:
                fi.write('vn %f %f %f\n' % (r[0], r[1], r[2]))

        if hasattr(mesh, 'ft'):
            for r in mesh.vt:
                if len(r) == 3:
                    fi.write('vt %f %f %f\n' % (r[0], r[1], r[2]))
                else:
                    fi.write('vt %f %f\n' % (r[0], r[1]))
        if hasattr(mesh, 'segm') and mesh.segm and not group:
            for p in mesh.segm.keys():
                fi.write('g %s\n' % p)
                for face_index in mesh.segm[p]:
                    write_face_to_obj_file(face_index, fi)
        else:
            if hasattr(mesh, 'f'):
                for face_index in range(len(mesh.f)):
                    write_face_to_obj_file(face_index, fi)


def write_mtl(mesh, path, material_name, texture_name):
    """Material attribute file serialization"""
    with open(path, 'w') as f:
        f.write('newmtl %s\n' % material_name)
        # copied from another obj, no idea about what it does
        f.write('ka 0.329412 0.223529 0.027451\n')
        f.write('kd 0.780392 0.568627 0.113725\n')
        f.write('ks 0.992157 0.941176 0.807843\n')
        f.write('illum 0\n')
        f.write('map_Ka %s\n' % texture_name)
        f.write('map_Kd %s\n' % texture_name)
        f.write('map_Ks %s\n' % texture_name)
