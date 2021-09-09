from PyQt5.QtGui import QIcon, QColor
from typings import Workflow, WorkflowObj
from typing import List

"""
WORKFLOW METADATA 
___________________
Array of JSON objects containing the following data:
    @name: 
    @type: ENUM type of Workflow
    @header: string displayed as "header"
    @desc: string displayed as "description" below header
    @checked: is workflow checked by default
    @graph: graph metadata:
        @type: type of graph
        @title: title of graph
        @x_label: x_label of graph
        @y_label: y_label of graph
        @x_type: the value measured on x axis
        @y_type: the value measured on y axis
    @props: array of optional parameters in the following format:
        @title: title of prop
        @placeholder: placeholder for prop label
"""
# TODO: ADD NEW WORKFLOW METADATA HERE
WORKFLOWS: List[WorkflowObj] = [
    {
    "name": "NND",
    "type": Workflow.NND,
    "header": "Nearest Neighbor Distance",
    "desc": "Find the nearest neighbor distance between gold particles. Optionally generate random coordinates.",
    "checked": True,
    "graph": {
        "type": "hist",
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
        "header": "Hierarchical Clustering",
        "desc": "Cluster gold particles into groups. Optionally generate random coordinates.",
        "checked": True,
        "graph": {
            "type": "bar",
            "title": "Hierarchical Clusters",
            "x_label": "# Particles In Cluster",
            "y_label": "# Of Clusters",
            "x_type": "cluster_id",
            "y_type": "cluster_id"
        },
        "props": [
            {
                "title": "distance_threshold (px)",
                "placeholder": "34"
            },
            {
             "title": "n_clusters",
             "placeholder": "None"
             },
            # {
            #  "title": "draw_clust_area",
            #  "placeholder": "False"
            #  },
        ]
    },
    {
        "name": "NND-CLUST",
        "type": Workflow.NND_CLUST,
        "header": "Nearest Neighbor Distance Between Clusters",
        "desc": "Find the nearest neighbor distance between clusters. Optionally generate random coordinates.",
        "checked": True,
        "graph": {
            "type": "hist",
            "title": "Nearest Neighbor Distance for Ward Hierarchical Clusters",
            "x_label": "Nearest Neighbor Distance",
            "y_label": "Number of Entries",
            "x_type": "dist",

        },
        "props": [
            {
                "title": "distance_threshold (px)",
                "placeholder": "34"
            },
            {"title": "n_clusters",
             "placeholder": "None"
             },
            {
                "title": "min_cluster_size",
                "placeholder": "3"
            },
        ]
    },
    {
        "name": "RIPPLER",
        "type": Workflow.RIPPLER,
        "header": "Gold Rippler: Spine-Particle Correlation",
        "desc": "Separate spine masks as individual components, grow components until they contain X gold particles, calculate Spine Correlated Particles Per P-face Area (SC3PA) (% of gold particles within spine masks) / (% of total area of p-face taken up by spine masks).  Requires lighthouse population.",
        "checked": False,
        "graph": {
            "type": "bar",
            "title": "Spine Correlated Particles Per P-face Area By Radius",
            "x_label": "radius",
            "y_label": "SC3PA",
            "x_type": "radius",
            "y_type": "SC3PA"
        },
        "props": [
            {
                "title": "max_steps",
                "placeholder": "10"
            },
            {
                "title": "step_size (px)",
                "placeholder": "60"
            },
        ]
    },
    {
        "name": "STARFISH",
        "type": Workflow.STARFISH,
        "header": "Starfish Nearest Neighbor Distance",
        "desc": "Find the nearest neighbor distance of two different populations. Requires lighthouse population.",
        "checked": False,
        "graph": {
            "type": "hist",
            "title": "Nearest Neighbor Distances of Different Pops",
            "x_label": "Nearest Neighbor Distance",
            "y_label": "Number of Entries",
            "x_type": "dist"
        },
        "props": []
    },
]

""" COLOR PALETTE OPTIONS """
PALETTE_OPS: List[str] = ["rocket", "crest", "mako", "flare", "viridis", "magma", "cubehelix", "rocket_r", "mako_r", "crest_r",
               "flare_r", "viridis_r", "magma_r", ]

""" METRIC UNIT OPTIONS """
UNIT_OPS: List[str] = ['px', 'nm', 'μm', 'metric']

""" UNIT SCALARS """
UNIT_PX_SCALARS = {
    'px': 1,
    'nm': 0.888,  # 1 px = 0.888nm
    'μm': 0.000888,  # 1px = 0.000888μm
    'metric': 1
}

""" NAVBAR ICON """
NAV_ICON = QIcon('foo.png')

""" MAX DIRS TO KEEP WHEN PRUNING OLD DIRS """
MAX_DIRS_PRUNE: int = 5

""" DEFAULT OUTPUT DIRECTORY """
DEFAULT_OUTPUT_DIR: str = './output'


""" PROGRESS BAR COLORS """

PROG_COLOR_1 = QColor(221, 221, 221) # 95, 197, 138) # # 240, 53, 218) # QColor(221, 221, 221)  # ddd
PROG_COLOR_2 = QColor(95, 197, 138) # 61, 217, 245)  # 233, 180, 33)  # 61, 217, 245) #QColor(233, 180, 33)  # e9b421
