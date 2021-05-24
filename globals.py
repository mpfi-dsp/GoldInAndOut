from PyQt5.QtGui import QIcon
from typings import Workflow

"""
WORKFLOW METADATA 
___________________
Array of JSON objects containing the following data:
    @type: ENUM type of Workflow
    @header: string displayed as "header"
    @desc: string displayed as "description" below header
    @hist: histogram metadata:
        @title: title of histogram
        @x_label: x_label of histogram
        @y_label: y_label of histogram
"""
WORKFLOWS = [{
    "name": "NND",
    "type": Workflow.NND,
    "header": "Nearest Neighbor Distance",
    "desc": "Find the nearest neighbor distance between gold particles. Optionally generate random coordinates.",
    "hist": {
        "title": "Distances Between Nearest Neighbors",
        "x_label": "Nearest Neighbor Distance",
        "y_label": "Number of Entries",
        "x_type": "dist"
    }
},
{
    "name": "CLUST",
    "type": Workflow.CLUST,
    "header": "Ward Hierarchical Clustering",
    "desc": "Cluster gold particles into groups. Optionally generate random coordinates.",
    "hist": {
            "title": "Ward Hierarchical Clusters",
            "x_label": "Cluster Value",
            "y_label": "Number of Entries",
            "x_type": "cluster"
        }
}
]

""" COLOR PALETTE OPTIONS """
PALETTE_OPS = ["rocket_r", "crest", "rocket", "mako", "flare", "viridis", "magma", "cubehelix", "mako_r",  "crest_r",  "flare_r", "viridis_r", "magma_r", ]

""" METRIC UNIT OPTIONS """
UNIT_OPS = ['px', 'nm', 'μm', 'metric']


""" NAVBAR ICON """
NAV_ICON = QIcon('foo.jpg')