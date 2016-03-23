#!/usr/bin/env python

import skimage.io as io
io.use_plugin('tifffile')

from skimage.filters import threshold_otsu, median, threshold_li
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import disk, square
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier

from skimage.morphology import closing, dilation, opening
from skimage.segmentation import clear_border


def load_movie(filename):
    movie = io.imread(filename)
    return movie


def movie_summary(movie, scaling_factor, filename, path):
    #load movie dimensions
    time, x_size, y_size = movie.shape
    sm_time = time//scaling_factor
    
    nrows = np.int(np.ceil(np.sqrt(sm_time)))
    ncols = np.int(sm_time//nrows+1)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows)) 
    count = 0
    for n in range(0, time, scaling_factor):
        i = count // ncols
        j = count % ncols
        axes[i, j].imshow(movie[n, ...], 
                        interpolation='nearest', 
                        cmap='gray')
        count += 1
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout()
    #save figure
    plt.savefig(path+'summary'+filename+'.png', format='png')
    plt.close()
    return

def thres_movie(movie, threshold_meth):
    max_int_proj = movie.max(axis=0)
    thresh_global = threshold_otsu(max_int_proj)
    return thresh_global

def smooth_movie(movie, smooth_size, smooth_meth, shape):
    smoothed_stack = np.zeros_like(movie)
    for z, frame in enumerate(movie):
        smoothed = smooth_meth(frame, shape(smooth_size))
        smoothed_stack[z] = smoothed    
    return smoothed_stack

def label_movie(movie, threshold, segmentation_meth, shape, size):
    labeled_stack = np.zeros_like(movie)
    for z, frame in enumerate(movie):
        im_max = frame.max()
        if im_max < threshold:
            labeled_stack[z] = np.zeros(frame.shape, dtype=np.int32)
        else:
            bw = segmentation_meth(frame > threshold, shape(size))
            cleared = bw.copy()
            clear_border(cleared)
            labeled_stack[z] = label(cleared)
    return labeled_stack
    

def segmentation_summary(movie, smoothed_stack, labeled_stack, scaling_factor, filename, path):
    time, x_size, y_size = movie.shape
    sm_time = time//scaling_factor
    
    nrows = np.int(np.ceil(np.sqrt(sm_time)))
    ncols = np.int(sm_time//nrows+1)
        
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(3*ncols, 1.5*nrows))
    count = 0
    for n in range(0, time, 55):
        i = count // ncols
        j = count % ncols * 2
        count += 1
        axes[i, j].imshow(smoothed_stack[n, ...], interpolation='nearest', cmap='gray')
        axes[i, j+1].imshow(labeled_stack[n, ...], interpolation='nearest', cmap='Dark2')
        
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j+1].set_xticks([])
        axes[i, j+1].set_yticks([])
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    
            
    fig.tight_layout()
    #save image!!!!
    plt.savefig(path+'seg'+filename+'.png', format='png')
    plt.close()
    return


def measure_properties(movie, labeled_stack):
    properties = []
    columns = ('x', 'y', 'time', 'I', 'A', 'I*A', 'radius')
    indices = []
    for z, frame in enumerate(labeled_stack):
        f_prop = regionprops(frame.astype(np.int), intensity_image = movie[z])
        for d in f_prop:
            radius = (d.area/np.pi)**0.5
            properties.append([d.weighted_centroid[0],
                            d.weighted_centroid[1],
                            z, d.mean_intensity, d.area,
                            d.mean_intensity * d.area,
                            radius])
            indices.append(d.label)
    if not len(indices):
        all_props = pd.DataFrame([], index=[])
    indices = pd.Index(indices, name='label')
    properties = pd.DataFrame(properties, index=indices, columns=columns)
    properties['I'] /= properties['I'].max()
    return properties

def labeled_summary(movie, scaling_factor,properties, labeled_stack, filename, path):
    time, x_size, y_size = movie.shape
    sm_time = time//scaling_factor
    
    nrows = np.int(np.ceil(np.sqrt(sm_time)))
    ncols = np.int(sm_time//nrows+1)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    
    count = 0
    for n in range(0, time, scaling_factor):
        plane_props = properties[properties['time'] == n]
        if not(plane_props.shape[0]) :
            continue
        i = count // ncols
        j = count % ncols
        count += 1
        axes[i, j].imshow(labeled_stack[n, ...],
                        interpolation='bicubic', cmap='Dark2')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        x_lim = axes[i, j].get_xlim()
        y_lim = axes[i, j].get_ylim()    
        
        
        axes[i, j].scatter(plane_props['y'], plane_props['x'],  
                        s=plane_props['I']*200, alpha=0.4)
        axes[i, j].scatter(plane_props['y'], plane_props['x'],
                        s=40, marker='+', alpha=0.4)
        axes[i, j].set_xlim(x_lim)
        axes[i, j].set_ylim(y_lim)       
    
    ## Remove empty plots 
    for ax in axes.ravel():
        if not(len(ax.images)):
                fig.delaxes(ax)  
            
    #fig.tight_layout()
    #figure save!!!!
    plt.savefig(path+'segAll'+filename+'.png', format='png')
    plt.close()
    return

def cluster_points(properties, max_dist):
    positions = properties[['x', 'y']].copy()

    dist_mat = dist.squareform(dist.pdist(positions.values))
    link_mat = hier.linkage(dist_mat)
    cluster_idx = hier.fcluster(link_mat, max_dist,
                                criterion='distance')
    properties['new_label'] = cluster_idx
    properties.set_index('new_label', drop=True, append=False, inplace=True)
    properties.index.name = 'label'
    properties = properties.sort_index()
    return properties

def df_average(df, weights_column):
    '''Computes the average on each columns of a dataframe, weighted
    by the values of the column `weight_columns`.
    
    Parameters:
    -----------
    df: a pandas DataFrame instance
    weights_column: a string, the column name of the weights column 
    
    Returns:
    --------
    
    values: pandas DataFrame instance with the same column names as `df`
        with the weighted average value of the column
    '''
    
    values = df.copy().iloc[0]
    norm = df[weights_column].sum()
    for col in df.columns:
        try:
            v = (df[col] * df[weights_column]).sum() / norm
        except TypeError:
            v = df[col].iloc[0]
        values[col] = v
    return values

def cluster_positions(properties):
    cluster_positions = properties.groupby(level='label').apply(df_average, 'I')
    return cluster_positions


def cluster_plot(movie, cluster_positions, properties, filename, path, xy_scale):
    time, x_size, y_size = movie.shape

    labels = cluster_positions.index.tolist()

    fig = plt.figure(figsize=(12, 12))
    colors = plt.cm.jet(properties.index.astype(np.int32))
    
    # xy projection:
    ax_xy = fig.add_subplot(111)
    ax_xy.imshow(movie.max(axis=0), cmap='gray')
    ax_xy.scatter(properties['y'],
                properties['x'],
                c=colors, alpha=0.2)
    
    
    ax_xy.scatter(cluster_positions['y'],
                cluster_positions['x'],
                c='r', s=50, alpha=1.)
    
    
    for i, txt in enumerate(labels):
        ax_xy.annotate(txt, (cluster_positions.iloc[i]['y'], cluster_positions.iloc[i]['x']))
    
    divider = make_axes_locatable(ax_xy)
    ax_yz = divider.append_axes("top", 2, pad=0.2, sharex=ax_xy)
    ax_yz.imshow(movie.max(axis=1), aspect=time/xy_scale, cmap='gray')
    ax_yz.scatter(properties['y'],
                properties['time'],
                c=colors, alpha=0.2)
    
    ax_yz.scatter(cluster_positions['y'],
                cluster_positions['time'],
                c='r', s=50, alpha=1.)
    
    
    ax_zx = divider.append_axes("right", 2, pad=0.2, sharey=ax_xy)
    ax_zx.imshow(movie.max(axis=2).T, aspect=xy_scale/time, cmap='gray')
    ax_zx.scatter(properties['time'],
                properties['x'],
                c=colors, alpha=0.2)
    
    ax_zx.scatter(cluster_positions['time'],
                cluster_positions['x'],
                c='r', s=50, alpha=1.)
    
    plt.draw()
    plt.savefig(path+'clust'+filename+'.png', format='png')
    plt.close()
    return

##testing code
#smooth_size = 1 # pixels
#min_radius = 2
#max_radius = 30
#t_scale = 1.0 #frame per second
#xy_scale = 0.106 #um per pixel
#
#
#path = fill
#filename = fill
#movie = fill
#movie_summary(movie, 55, filename, path)
#threshold = thres_movie(movie, threshold_otsu)
##smoothed_movie = smooth_movie(movie, smooth_size, median, disk)
#labeled_movie = label_movie(movie, threshold, opening, disk, 1)
#segmentation_summary(movie, movie, labeled_movie, 11, filename, path)
#properties = measure_properties(movie, labeled_movie)
#labeled_summary(movie, 11,properties, labeled_movie, filename, path)
#properties_clust = cluster_points(properties, 25)
#cluster_pos = cluster_positions(properties_clust)
#cluster_plot(movie, cluster_pos, properties_clust, filename, path, xy_scale)