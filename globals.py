from PyQt5.QtGui import QIcon
from typings import Workflow

"""
WORKFLOW METADATA 
___________________
Array of JSON objects containing the following data:
    @name: 
    @type: ENUM type of Workflow
    @header: string displayed as "header"
    @desc: string displayed as "description" below header
    @graph: histogram metadata:
        @title: title of histogram
        @x_label: x_label of histogram
        @y_label: y_label of histogram
    @props: array of optional parameters in the following format:
        @title: title of prop
        @placeholder: placeholder for prop label
"""
WORKFLOWS = [{
    "name": "NND",
    "type": Workflow.NND,
    "header": "Nearest Neighbor Distance",
    "desc": "Find the nearest neighbor distance between gold particles. Optionally generate random coordinates.",
    "graph": {
        "title": "Distances Between Nearest Neighbors",
        "x_label": "Nearest Neighbor Distance",
        "y_label": "Number of Entries",
        "x_type": "dist"
    },
    "props": []
},
{
    "name": "CLUST",
    "type": Workflow.CLUST,
    "header": "Ward Hierarchical Clustering",
    "desc": "Cluster gold particles into groups. Optionally generate random coordinates.",
    "graph": {
            "title": "Ward Hierarchical Clusters",
            "x_label": "Cluster ID",
            "y_label": "Cluster Size",
            "x_type": "cluster_id"
        },
    "props": [
            {
            "title": "distance_threshold",
              "placeholder": "120"
            },
            {"title": "n_clusters",
             "placeholder": "None"
             },
        ]
},
{
    "name": "NND-CLUST",
    "type": Workflow.NND_CLUST,
    "header": "Nearest Neighbor Distance Between Clusters",
    "desc": "Find the nearest neighbor distance between clusters. Optionally generate random coordinates.",
    "graph": {
            "title": "Nearest Neighbor Distance for Ward Hierarchical Clusters",
            "x_label": "Nearest Neighbor Distance",
            "y_label": "Number of Entries",
            "x_type": "dist"
        },
    "props": [
        {
        "title": "distance_threshold",
          "placeholder": "120"
        },
        {"title": "n_clusters",
         "placeholder": "None"
         },
        {
            "title": "min_cluster_size",
            "placeholder": "3"
        },
        ]
}
]

""" COLOR PALETTE OPTIONS """
PALETTE_OPS =  ["rocket", "crest",  "mako", "flare", "viridis", "magma", "cubehelix", "rocket_r", "mako_r",  "crest_r",  "flare_r", "viridis_r", "magma_r", ]

""" METRIC UNIT OPTIONS """
UNIT_OPS = ['px', 'nm', 'μm', 'metric']


""" NAVBAR ICON """
NAV_ICON = QIcon('foo.png')


""" MAX DIRS TO KEEP WHEN PRUNING OLD DIRS """
MAX_DIRS_PRUNE = 5