from psbody.mesh import Mesh
import numpy as np
import torch
import igl

def smoothness_obj(L, verts):
    # Sadly there is no torch.sparse.blkdiag method yet, so we need to do
    obj_list = []
    for i in range(0,3):
        x = verts[:,i].unsqueeze(0).t()
        xtLx = torch.mm(x.t(), torch.mm(L,x))
        obj_list.append(xtLx)
    obj =  torch.stack(obj_list, dim=0).sum(dim=0).sum(dim=0)[0]
    return obj

def smoothness_obj_from_ref(L, verts, verts_0):
    # Sadly there is no torch.sparse.blkdiag method yet, so we need to do
    obj_list = []
    for i in range(0,3):
        x_sub_x0 = verts[:,i].unsqueeze(0).t()-verts_0[:,i].unsqueeze(0).t()
        smoothness_diff_x = torch.mm(x_sub_x0.t(), torch.mm(L,x_sub_x0))
        obj_list.append(smoothness_diff_x)
    obj =  torch.stack(obj_list, dim=0).sum(dim=0).sum(dim=0)[0]
    return obj

def smoothness_obj_coord(L,coord):
    vt = coord.unsqueeze(0)
    Lv = torch.mm(L, vt.t())
    vtLv = torch.mm(vt, Lv)
    return vtLv

def torch_laplacian_cot(verts_np,faces_np):
    L_np = -1*igl.cotmatrix(verts_np, faces_np)
    return coo_to_sparse_torch(L_np.tocoo())

def coo_to_sparse_torch(mat, dtype = torch.float32):

    # tensor flow sparse
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape
    return torch.sparse_coo_tensor(i, v, dtype=dtype)

if __name__ == '__main__':

    # A set of sanity checks (invariant to translation) for the cotan laplacian
    target_mesh = Mesh(filename='../Results/imgHQ00039.obj')
    v,f = target_mesh.v, np.int32(target_mesh.f)
    L = -1*igl.cotmatrix(v, f).tocoo()

    x = np.expand_dims(v[:,0], axis=1)
    def smooth_np(L,x):
        return x.transpose().dot( L.dot(x) )

    def smooth_diff_np(L,x1,x2):
        return smooth_np(L,x1-x2)

    print ('smooth_np(L,v[:,0]) = ', smooth_np(L,x))
    print ('smooth_diff_np(L,x,x) = ', smooth_diff_np(L,x,x))
    x_translated = x + 100000
    print ('smooth_diff_np(L,x,x) = ', smooth_diff_np(L,x,x_translated))


    # now sanity checks on tensors

    tensor_v = torch.tensor(target_mesh.v)
    tensor_f = torch.tensor(np.int32(target_mesh.f), dtype=torch.long)#torch.from_numpy(np.long(target_mesh.f))#torch.tensor(target_mesh.f, dtype=torch.int32)

    print ('tensor_v.unsqueeze(0) = ', tensor_v.unsqueeze(0).shape)


    L = torch_laplacian_cot(v, f)
    print ('L.shape = ', L.shape)
    
    print ('tensor_v.dtype = ', tensor_v.dtype)
    print ('L.dtype = ', L.dtype)
    smooth_obj = smoothness_obj(L,tensor_v)

    print ('smooth_obj = ', smooth_obj)

    ones = torch.ones(tensor_v.shape, dtype=torch.float32)
    smooth_obj = smoothness_obj(L,ones)
    print ('smooth_obj = ', smooth_obj)

    smooth_obj = smoothness_obj(L,tensor_v+ones)
    print ('smooth_obj = ', smooth_obj)

    smooth_obj_from_ref = smoothness_obj_from_ref(L,tensor_v, tensor_v+ones)
    print ('smooth_obj_from_ref = ', smooth_obj_from_ref)