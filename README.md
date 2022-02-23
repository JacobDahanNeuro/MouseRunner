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

### Adjusting In-Built (Hard-Coded) Parameters
> Note that these adjustments _must_ be made prior to generating a desktop executable.

Some parameters, such as the [COM port](https://en.wikipedia.org/wiki/COM_(hardware_interface)) for Arduino communication may not align with in-built defaults. These can be easily adjusted within the Python scripts, but are not available as input parameters to simplify GUI interactions. Of note, please keep careful mind of the following parameters:
- `time.sleep(n)` : Prior to transferring files, a delay of `n` seconds is provided to allow for completion of image compilation. This is handled in the script `MouseRunner`. With high-throughput data ports and a fast computer, `n` can be set to zero, such that there is no delay (default). Slower computers or USB hubs will require longer delays, particularly for longer behavioral sessions with buffer build-up.
- `COMX` : During Arduino initiation, a COM port must be assigned for communication. This is handled in the script `MouseRunner`. The current port in use can be determined via the [Arduino IDE](https://support.arduino.cc/hc/en-us/articles/4406856349970-Find-the-port-your-board-is-connected-to) or, on Windows, via the [Device Manager](https://www.mathworks.com/help/supportpkg/arduinoio/ug/find-arduino-port-on-windows-mac-and-linux.html).

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
- [ ] Document MouseRunner
- [ ] Develop `find_arduino` function for automatic handling of COM port changes
- [ ] Add instructions for Fresco Drive bandwidth modifications
