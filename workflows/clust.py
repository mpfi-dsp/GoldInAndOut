from sklearn.cluster import AgglomerativeClustering
"""
WARD HIERARCHICAL CLUSTERING
_______________________________
@data: dataframe with coordinates scaled to whatever format desired
@prog_wrapper: progress bar wrapper element, allows us to track how much time is left in process
@gen_rand: generate random coordinates
@img_path: path to image we are finding the n nearest distance of (only needed if gen_rand is True)
@pface_path: path to mask we are finding the n nearest distance of (only needed if gen_rand is True)
@n_rand_to_gen: number of random particles to generate
"""
def run_clust(data, distance_threshold=0, n_clusters=None, affinity='euclidean', linkage='ward'):
    print(data.head())
    hc = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    print(hc)