from PyQt5.QtGui import QIcon, QColor
from typings import Workflow, WorkflowObj
from typing import List


""" VERSION NUMBER """
VERSION_NUMBER: str = '1.18.1'

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
        "header": "Nearest Neighbor Distance of Particles",
    "desc": "Find the nearest neighbor distance (NND) between gold particles. Optionally generate random coordinates.",
    "checked": True,
    "graph": {
        "type": "hist",
        "title": "NND of Particles",
        "x_label": "Distance",
        "y_label": "Number of Particles",
        "x_type": "dist"
    },
    "props": []
},
    {
        "name": "CLUST",
        "type": Workflow.CLUST,
        "header": "Hierarchical Clustering of Particles",
        "desc": "Cluster gold particles into groups. Optionally generate random coordinates.",
        "checked": True,
        "graph": {
            "type": "bar",
            "title": "Hierarchical Clustering of Particles",
            "x_label": "Number of Particles In Cluster",
            "y_label": "Number of Clusters",
            "x_type": "cluster_id",
            "y_type": "cluster_id"
        },
        "props": [
            {
                "title": "distance threshold (px)",
                "placeholder": "27"
            }
        ]
    },
    {
        "name": "SEPARATION",
        "type": Workflow.SEPARATION,
        "header": "Separation Between Clusters",
        "desc": "Find the separation (NND) between clusters. Optionally generate random coordinates.",
        "checked": True,
        "graph": {
            "type": "hist",
            "title": "Separation Between Clusters",
            "x_label": "Nearest Neighbor Distance",
            "y_label": "Number of Entries",
            "x_type": "dist",

        },
        "props": [
            {
                "title": "distance threshold (px)",
                "placeholder": "27"
            },
            {
                "title": "minimum clust size",
                "placeholder": "2"
            },
        ]
    },
    {
        "name": "RIPPLER",
        "type": Workflow.RIPPLER,
        "header": "Gold Rippler: Landmark-Particle Correlation",
        "desc": "Separate landmark masks as individual components, grow components until they contain X gold particles, calculate Landmark Correlated Particle Index (LCPI): (% of gold particles within landmark masks) / (% of total area of img taken up by landmark masks).  Requires lighthouse population.",
        "checked": False,
        "graph": {
            "type": "bar",
            "title": "Landmark Correlated Particle Index (LCPI) By Radius",
            "x_label": "radius",
            "y_label": "LCPI",
            "x_type": "radius",
            "y_type": "LCPI"
        },
        "props": [
            {
                "title": "maximum steps",
                "placeholder": "10"
            },
            {
                "title": "step size (px)",
                "placeholder": "60"
            },
            {
                "title": "initial radius (px)",
                "placeholder": "50"
            }
        ]
    },
    {
        "name": "GOLDSTAR",
        "type": Workflow.GOLDSTAR,
        "header": "Gold Star Nearest Neighbor Distance",
        "desc": "Find the nearest neighbor distance of two different populations. Requires lighthouse population",
        "checked": False,
        "graph": {
            "type": "hist",
            "title": "Gold Star NND",
            "x_label": "Distance",
            "y_label": "Number of Particles",
            "x_type": "dist"
        },
        "props": [
            {
                "title": "A* (around landmarks)",
                "placeholder": "0"
            }
        ]
    },
]

""" COLOR PALETTE OPTIONS """
PALETTE_OPS: List[str] = ["rocket", "crest", "mako", "flare", "viridis", "magma", "cubehelix", "rocket_r", "mako_r", "crest_r",
 "flare_r", "viridis_r", "magma_r", ]

REAL_COLOR = (212, 118, 41)
RAND_COLOR = (62, 193, 213)

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

""" RANDOM COORDINATES DEFAULT DISTANCE THRESHOLD (px) """
DEFAULT_DISTANCE_THRESH = 5

""" PROGRESS BAR COLORS """

PROG_COLOR_1 = QColor(221, 221, 221)
PROG_COLOR_2 = QColor(0, 190, 204)
