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

## Usage

### Installing FFMPEG

Video writing to mp4 requires manual FFMPEG installation. To install FFMPEG on Windows, simply follow the instructions [here](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/).

> Note that FFMPEG downloads are zipped by default and will require accessory programs such as [7zip](https://www.7-zip.org/download.html) for installation.

To test that the FFMPEG installation was successful, open the [Windows powershell](https://docs.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.2) and enter `ffmpeg`.

> Note that Windows operating systems do not include mp4 codecs by default. These may or may not require installation via [third parties](http://codecguide.com/download_k-lite_codec_pack_basic.htm).

### Doric Neuroscience Studio

Arduino-based laser triggers are intended for use with [Doric Neuroscience Studio](https://neuro.doriclenses.com/products/doric-neuroscience-studio), which can command numerous laser wave patterns with simple `On` and `Off` Arduino signals. Installation the Studio is not required, though the code must be adapted to include a wave generator for complex laser commands.
> Note that any code modifications must be made _prior_ to building the Desktop Executable.
>
> Doric Neuroscience Studio refers to rising and falling edges as `gates` rather than `triggers`, and the configuration must be set up accordingly. See the built-in `haloTrigger` Doric configuration file for more information.

After installing the Studio from the above link, simply open the application and set up a custom configuration, `MouseRunner` will handle the rest.

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

## To-Do

- [ ] Incorporate [DeepLabCut Live](https://github.com/DeepLabCut/DeepLabCut-live)
- [ ] Document [MouseRunner](https://github.com/JacobDahanNeuro/MouseRunner/blob/main/scripts/behavior/MouseRunner.py)
- [x] Develop `find_arduino` function for automatic handling of [COM port changes](https://stackoverflow.com/questions/24214643/python-to-automatically-select-serial-ports-for-arduino)
- [ ] Add instructions for [Fresco Drive](https://www.flir.com/support-center/iis/machine-vision/knowledge-base/usb-3.1-cameras-with-fresco-driver-limited-to-6-mb) [bandwidth modifications](http://www.uninstallhelps.com/how-to-uninstall-fresco-logic-usb3-0-host-controller.html) 
- [x] Add [Doric Neuroscience Studio](https://neuro.doriclenses.com/products/doric-neuroscience-studio) installation instructions
