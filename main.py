import sys
import os
import numpy as np

import SimpleITK as sitk
from multiprocessing import Pool
import scipy.ndimage as ndi
import networkx as nx
import imageio
import matplotlib

def get_bboxes(sitkimage):
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(sitkimage)
    label_values = shape_stats.GetLabels()
    bboxes = [shape_stats.GetBoundingBox(label_value) for label_value in label_values]
    return bboxes,label_values

def crop(bbox, label_arr):
    shape = label_arr.shape
    return label_arr[max(0,bbox[2]-1):min(shape[0],bbox[2]+bbox[5]+1), 
                     max(0,bbox[1]-1):min(shape[1],bbox[1]+bbox[4]+1), 
                     max(0,bbox[0]-1):min(shape[2],bbox[0]+bbox[3]+1)]

def flatten_array(nested_array):
    flattened = []
    for element in nested_array:
        if isinstance(element, list):
            flattened += flatten_array(element)
        else:
            flattened.append(element)
    return flattened

def contact_surface_pixels(particles, label_value):
    bmsk = particles==label_value
    dilated = ndi.binary_dilation(bmsk)
    dilated[bmsk] = 0
    neighbor_labels = particles[dilated]
    neighbor_labels = neighbor_labels[neighbor_labels>0]
    return [(label_value,neighbor_label) for neighbor_label in neighbor_labels]

def construct_graph(touching):
    G = nx.Graph()
    for edge in np.unique(touching,axis=0):
        G.add_edge(edge[0],edge[1],weight=0)
    for edge in touching:
        G.edges[edge]['weight'] += 1
    return G

def threshold_graph_edge_weight(G, threshold_value):
    for edge in G.edges:
        if G.edges[edge]['weight']<threshold_value:
            G.remove_edge(edge[0],edge[1])# remove edges whose weight is smaller than a threshold value

    dict_cc = {}# cc: connected component in the graph
    # if one connected component is (1,2,5), add {2:1, 5:1} to dict_cc to indicate that label 2 is relabeled as 1 and label 5 is also relabeled as label 1
    for cc in list(nx.connected_components(G)):# for each connected component
        if len(cc)==1:
            continue# no merge needed if the cc only contains one label
        cc_min = min(cc)# choose the min value of all labels in the cc as the final label after merge
        cc_non_min = cc-set([cc_min])# all labels in the cc except for the chosen label
        for ele in cc_non_min:
            dict_cc[ele] = cc_min

    # if label only have 6 labels np.array([0,1,2,3,4,5]), {2:1, 5:1} generates np.array([0,1,1,3,4,1])
    arr_cc = np.zeros(label_values[-1]+1,dtype=np.uint32)
    for label_value in label_values:
        if label_value in dict_cc.keys():
            arr_cc[label_value] = dict_cc[label_value]
        else:
            arr_cc[label_value] = label_value
    return arr_cc


def saveMP4(MP4_path,arr3d,fps=20):
    """save imagestack as MP4 file with `imageio <https://pypi.org/project/imageio/>`_
    
    Parameters
    ----------
    MP4_path : str
        path to save the output MP4 file
    arr3d : np.ndarray
        slices of rgb images (SXYC)
    fps : int
        frames per second

    """
    writer = imageio.get_writer(MP4_path, fps=fps)
    for z in range(arr3d.shape[0]):
        writer.append_data(arr3d[z])
    writer.close()
    print('Saved '+MP4_path)

class RandomColormap:
    """
    apply random colormap to a label image
    """
    def __init__(self,num_features):
        np.random.seed(0)
        self.cmap = matplotlib.colors.ListedColormap(np.vstack([[[1,1,1],],np.random.rand(num_features-1,3)]))
        self.norm = matplotlib.colors.Normalize(vmin=0,vmax=num_features)
        self.scalar_mappable = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    def toRGB(self,arr):# accepts 2d gray scale image 
        return self.scalar_mappable.to_rgba(arr,bytes=True)[...,:3]
    def toRGB_by_slice(self,arr3d):# accepts 3d gray scale image
        assert hasattr(arr3d,'ndim') and arr3d.ndim==3, "only accept 3D array"
        return np.stack([self.toRGB(arr3d[z]) for z in np.arange(arr3d.shape[0])],0)

if __name__ == "__main__":
    cwd = os.getcwd()
    particles_sitk = sitk.ReadImage('D3459-half-dlseg_ws_crop'+'.mhd')
    particles_arr = sitk.GetArrayFromImage(particles_sitk)

    bboxes,label_values = get_bboxes(particles_sitk)
    cropped_lst = [[crop(bbox,particles_arr),label_value] for bbox,label_value in zip(bboxes,label_values)]
    with Pool() as pool:
        touching = pool.starmap(contact_surface_pixels, cropped_lst)
    touching = flatten_array(touching)
    G = construct_graph(touching)
    particles_arr_thresh3000 = threshold_graph_edge_weight(G.copy(), 3000)[particles_arr]

    stacked = np.dstack([particles_arr,particles_arr_thresh3000])# stack horizontally
    wsrgb = RandomColormap(stacked.max()).toRGB_by_slice(stacked)
    saveMP4('ws_threshold_contact_surface_thresh3000.mp4',wsrgb)