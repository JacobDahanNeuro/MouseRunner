# MouseRunner by @JacobDahanNeuro

## Getting Started


### Setting Up a Virtual Environment (recommended)

> Note that specific Python version requirements (<= 3.8) may require the creation of a virtual conda environment in order to access Spinnaker controls.

After [installing Anaconda](https://www.anaconda.com/products/individual), open the [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/). This kernel will enable you to create a virtual environment, install requisite packages, and run the MouseRunner GUI.

Within the Prompt, enter the following commands:

```
conda create -n "MouseRunner" python=3.8    # Create new environment MouseRunner with default python version 3.8
conda activate MouseRunner                  # Activate the new environment
```

> Note that Spinnaker cameras are only operable via Python versions 3.8 or below.
> 
> For interactive capabilities, enter `conda create -n "MouseRunner" python=3.8 ipython` instead.

### Installing Packages
All required basic Python modules can be found within the [requirements document](requirements.txt) and should be installed within the virual environment by entering the following command in the Prompt: 

```pip install -r /path/to/requirements.txt```.

The Spinnaker software module PySpin must be installed manually according to the specifications of your computer's OS from [Spinnaker's own website](https://www.flir.eu/products/spinnaker-sdk/).

> Note that additional installation instructions can be found within the software's README file, included in the software download.

### Building A Desktop Executable (optional)

To generate a desktop executable for simple click-and-run functionality of MouseRunner, enter the following command (adjusted for your save location):

```
cd /path/to/MouseRunner/scripts/behavior
pyinstaller.exe --onefile --console --icon=/path/to/MouseRunner/assets/icon.ico /path/to/MouseRunner/scripts/behavior/MouseRunner.py
```

> On Windows operating systems, change directories using the command `cd /d /path/to/MouseRunner/scripts/behavior`
> 
> Note that this command will take some time.

After the operation is completed, move to the newly created directory `dist`, create a shortcut for the executable in the folder, and move the shortcut to the Desktop. 