# Gold In-and-Out 

🔬🥇🧠🍔 Automated Gold Particle Analysis For Freeze Fracture Replica Electron Microscopy Images

<!-- Read more about Gold In-and-Out in the paper *"GIO: A Toolkit for Analyzing Subcellular Distribution of Immunogold-Labeled Membrane Proteins in Freeze-Fracture Replica Images"* published [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ)! -->

*Note: The terms analysis `method` and `workflow` are used relatively interchangeably in this readme.*

<img src="https://user-images.githubusercontent.com/47064842/133366931-875c5c01-fd15-4993-a656-0874e7f83b3b.png" width="500px">

## Design 

### Package Overview

Gold In-and-Out (GIO) is built on PyQt5, a powerful python graphical user interface framework responsible for other amazing scientific software such as ORS Dragonfly.

In general, programming design choices were made with the goals of creating readable, extensible, and reliable code. Descriptive variable names and a heavy use of comments allow for this, as well as handling exceptions in almost any case where a crash is possible, and avoiding repeating code unnecessarily. I have also made use of the python equivalent of types and interfaces whenever possible. 

On a broad level, Gold InAndOut was designed to combine numerous analysis workflows and simplify the development process of adding additional methods. The code for GoldInAndOut is split among two parts, graphical interface elements and data science analysis. 

### Graphical Interface

All of the graphical interface code is located in the `/views` directory. This includes the primary window interface as well as the powerful built-in image viewer and logger. 

All code for the primary window is contained in two files, `home.py` and `workflow.py`. This allows us to dramatically reduce excess or repeated code by generating different variations of the same core workflow analysis view. The home view lets the user input their desired input and global parameters, and run analysis. This will then populate the navigation sidebar with all selected workflow pages, which the user can toggle between at will for further customization and analysis.


### Data Science Analysis

All data science analysis workflow files and related functions are contained in the `/workflows` directory. Each file is named after its respective workflow, and contains two functions: one that "runs" the workflow and outputs the resulting data, and one that takes that data and generates output visualizations. These are all run simultaneously using multithreading, speeding up each run of GoldInAndOut. Many of these workflow methods take custom parameters, which are passed from the external thread. They also emit updates to a progress bar at intermittent points throughout their run. 

There are five analysis methods included in the base version of GoldInAndOut:
- Nearest Neighbor Distance
- Hierarchical Clustering
- Separation Between Clusters
- Gold Rippler: Landmark-particle Analysis
- Gold Star Nearest Neighbor Distance

To learn more about these workflows, check the [wiki](https://github.com/GoldinGuy/GoldInAndOut/wiki/Workflows)

<img src="https://user-images.githubusercontent.com/47064842/137605418-3488526a-4b83-485a-a527-0162538c47e1.png" width="500px">

## Development

I recommend using either a virtual environment or some other python package manager such as Anaconda to develop or test this package. You must have all required dependencies available locally to test or debug GoldInAndOut. However, the production build is a one-file solution with no complex installation or package management necessary.

PyQt5 is incredibly verbose (with a simple button requiring up to 6 lines of code). I recommend reading about the framework to familiarize yourself with some of the syntax before diving in. The docs can be found [here](https://doc.qt.io/qtforpython/#documentation). Familiarity with numpy, pandas, scikit, and other popular python data science packages is also recommended.

### Installation

Clone this repository to your desired location, then run the following commands:

```python
cd GoldInAndOut

pip install -r requirements.txt

python -u main.py
```

### Adding Additional Custom Workflows

Gold In-and-Out was specifically designed to make it incredibly easy to add new analysis methods to the package. You don't need to add a single line of code to either of the interface files to have a functional analysis workflow! 

To add a new method, follow these four simple steps:
- Add it as a typing
- Code the actual analysis
- Add it to globals
- Thread it! 

1) Add it as a typing

In the `typings.py` file, add your workflow to the `Workflow` Enum. The enum should appear as the following:

```python
class Workflow(Enum):
    NND = 1
    CLUST = 2
    ...
    YOUR_WORKFLOW = (LAST NUMBER + 1) 
```
You will use this as a reference elsewhere in the project. 

2) Code the actual analysis

Add a python file to the `workflows/` directory containing the workflow or macro you'd like to add to the GIO GUI. Ideally, this is in the format `<WorkflowName>.py`.

Your file should contain at minimum a "run" method that takes in the real and random coordinate lists (in the format `List[Tuple[float, float]]`) (and optionally the progress bar event emitter `pyqtSignal`), performs your analysis math, and returns two pandas dataframes (one for real, one for random). You can also take the image, mask, or csv files, or any other custom parameters you require.

You should also include a method that takes in this data to create some form of visualization on the image, for example, for nearest neighbor distance a function that draws the coordinates and lines connecting their nearest distances on the image using opencv.

<img src="https://user-images.githubusercontent.com/47064842/133363622-485776be-f0a7-4e09-9b6b-79b6546066e0.png" width="500px">


3) Add it to globals

In the `globals.py` file, add your workflow to the `WORKFLOW_METADATA` dictionary. This is in the following format:

```s
    @name: short abbreviation of workflow, no spaces
    @type: ENUM type of Workflow
    @header: string displayed as "header"
    @desc: string displayed as "description" below header
    @hist: histogram metadata:
        @title: title of histogram
        @x_label: x_label of histogram
        @y_label: y_label of histogram
    @props: array of optional parameters in the following format:
        @title: title of prop
        @placeholder: placeholder for prop label
```

Below is a sample workflow using JSON structure:

```js
{
    "name": "CLUST",
    "type": Workflow.CLUST,
    "header": "Hierarchical Clustering",
    "desc": "Cluster gold particles into groups. Optionally generate random coordinates.",
    "hist": {
            "title": "Hierarchical Clusters",
            "x_label": "Cluster Value",
            "y_label": "Number of Entries",
            "x_type": "cluster"
        },
    "props": [
            {
              "title": "distance threshold (px)",
              "placeholder": "27"
            },
            {
             "title": "number of clusters",
             "placeholder": "None"
            }
        ]
}
```
This workflows object is looped through to initialize the main page and is what allows us to not need to write any custom ui code for any particular workflow. If you need custom parameters for your workflow, you can add them here to the `props` section. These will automatically appear on their respective workflow page as input fields that pass strings down to your method function if you need them. 

4) Thread it!

Finally, in the `threads.py` file, under label `ADD NEW WORKFLOWS HERE`, add an `elif` statement for your workflow that reads similar to the following:

```python
 if wf['type'] == Workflow.NND:
    real_df1, rand_df1 = run_nnd(real_coords=coords, rand_coords=rand_coords, pb=self.progress)
elif wf['type'] == Workflow.CLUST:
    real_df1, rand_df1, real_df2, rand_df2 = run_clust(real_coords=coords, rand_coords=rand_coords, img_path=img_path, distance_threshold=vals[0], n_clusters=vals[1], pb=self.progress, clust_area=clust_area)
```
If your workflow requires custom parameters/props, use custom values in the format `vals[0]`, `vals[1]`, `vals[2]`, etc. These represent the custom props in the globals json file following that order. This will appear in the following format in `workflow.py`:

```python
    def get_custom_values(self):
        return [self.cstm_props[i].text() if self.cstm_props[i].text() else self.wf['props'][i]['placeholder'] for i in range(len(self.cstm_props))]
```

Graphs will be automatically generated based on your input in the globals file. An example is below:

<img src="https://user-images.githubusercontent.com/47064842/133366982-91610e29-51d2-4166-a2f1-94fde45258bf.png" width="500px">


If you want to use your custom visualization method, you will also need to add that to the `workflow.py` file under `ADD NEW VISUALIZATIONS HERE` in the same vein by adding an elif statement with your workflow. For example, for the "seperation between clusters" analysis method:

```python
 elif wf["type"] == Workflow.SEPARATION:
    vals = self.get_custom_values()
    if self.gen_real_cb.isChecked():
        drawn_img = draw_separation(nnd_df=self.data.real_df1, clust_df=self.data.real_df2, img=drawn_img,
                                    palette=palette, bin_counts=n, circle_c=(103, 114, 0),  distance_threshold=vals[0], draw_clust_area=self.draw_clust_area)
    if self.gen_rand_cb.isChecked():
        drawn_img = draw_separation(nnd_df=self.data.rand_df1, clust_df=self.data.rand_df2, img=drawn_img,
                                    palette=r_palette, bin_counts=n, circle_c=(18, 156, 232), distance_threshold=vals[0], draw_clust_area=self.draw_clust_area)
```

<img src="https://user-images.githubusercontent.com/47064842/137605445-66f459d0-90c4-4acb-b569-05f8ce1db838.png" width="500px">


### Testing Gold In-and-Out

You can run Gold In-and-Out locally by either typing `python -u main.py` in the terminal or using a text editor or IDE with a debugger. 

```powershell
[Running] python -u "c:\Users\yourname\...\GoldInAndOut\main.py"
INFO:root:Booting up...
INFO:root:Detecting cores...
INFO:root:Building layout...
INFO:root:Initializing main window...
```

For a text editor, I recommend [Visual Studio Code](https://code.visualstudio.com/) with python installed and the [Code Runner Extension](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner). For an all-in-one solution, [PyCharm](https://www.jetbrains.com/pycharm/) is a fantastic integrated development environment.

### Compiling To Executable

We are using [pyinstaller](https://www.pyinstaller.org/#)) to compile Gold In-and-Out into a finished application. The steps differ based on your platform, but the following instructions are for windows (10-11).

First, ensure you have a `main.spec` and `resources.qrc` file. If you don't, run the following commands:

```
pyinstaller main.spec

pyrcc5 resources.qrc -o resources.py
```

Then to compile to exe enter the directory with `main.py` and run the following command:

WINDOWS:
```
pyinstaller.exe --onefile --windowed --icon=logo.ico main.py
```

MACOS:
```
pyinstaller --onefile --windowed --icon=logo.ico --codesign-identity=YOUR_APPLE_DEVELOPER_CERT_CODESIGN_IDENTITY --target-arch [x86_64|universal2] main.py
```

This will spew output in the console and may take a while. The final exe file can be found in the `/dist` directory.

I recommend reading through this handy article for more details regarding the specifics of compilation, particularly if you're having trouble getting icons to load: [Packaging PyQt5 applications for Windows, with PyInstaller](https://www.pythonguis.com/tutorials/packaging-pyqt5-pyside2-applications-windows-pyinstaller/)

*Pyinstaller should be capable of building for MacOS, but I have yet to test it. If you're trying to compile for Linux, you're capable of figuring out how (or you can just use wine).*

## References & Thanks

Created for the Electron Microscopy Core of the Max Planck Florida Institute For Neuroscience.

Thanks to all those who helped with this research project, particularly my mentors at the Max Planck Institute for Neuroscience including Dr. Diego Jerez, Dr. Naomi Kamasawa, Dr. Debbie Guerrero-Given, Connon Thomas, Dr. Matthias Haury, and Dr. Joe Schumacher.

Additional shoutout to Stack Overflow, the greatest friend any programmer could ever ask for.

