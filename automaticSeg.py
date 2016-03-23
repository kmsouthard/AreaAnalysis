import os
import fnmatch
from segmentationFunctions import *

path = "your path here"

pattern = '*.tif'
smooth_size = 1 # pixels
min_radius = 2
max_radius = 30
t_scale = 1.0 #frame per second
xy_scale = 0.106 #um per pixel


i = 0
for (path, dirs, files) in os.walk(path):
    for filename in fnmatch.filter(files, pattern):
        print filename
        movie = load_movie(os.path.join(path, filename))
        movie_summary(movie, 55, filename, path)
        threshold = thres_movie(movie, threshold_otsu)
        smoothed_movie = smooth_movie(movie, smooth_size, median, disk)
        labeled_movie = label_movie(movie, threshold, opening, disk, 1)
        segmentation_summary(movie, movie, labeled_movie, 11, filename, path)
        properties = measure_properties(movie, labeled_movie)
        labeled_summary(movie, 11,properties, labeled_movie, filename, path)
        properties_clust = cluster_points(properties, 30)
        cluster_pos = cluster_positions(properties_clust)
        cluster_plot(movie, cluster_pos, properties_clust, filename, path, xy_scale)
        properties_clust.to_csv(path+filename+'.csv')
        i += 1
        if i > 10:
            break