<p align="center">
  <img width="150" src="./Images/KathleenWang_for_ezTrack.png">
</p>

# Behavior Tracking with ezTrack (fork by davor)

### About this fork
This is a fork of the original ezTrack repository, featuring enhanced batch processing, programmatic ROI definitions, and performance improvements. If you encounter problems or have questions or feature requests, feel free to open an issue here on GitHub, email me, or **find me on [TheBehaviourForum](https://www.thebehaviourforum.org/)!**

By the way, **[TheBehaviourForum](https://www.thebehaviourforum.org/)** is a great place to discuss behavioral analysis tools and methods, and to get help from the community. Join the conversation!

<a href="https://www.thebehaviourforum.org/"><img src="Images/TheBehaviourForum.png" height="40"></a>

### About ezTrack
ezTrack provides Jupyter notebooks and Python utilities to track animal location, motion, and freezing from video recordings. The toolkit is organized into two modules: Location Tracking and Freeze Analysis.

If you are new to Jupyter, see **[Getting Started](https://github.com/DeniseCaiLab/GettingStarted)**.

# Please cite ezTrack if you use it in your research:
Pennington ZT, Dong Z, Feng Y, Vetere LM, Page-Harley L, Shuman T, Cai DJ (2019). ezTrack: An open-source video analysis pipeline for the investigation of animal behavior. Scientific Reports: 9(1): 19979

# Check out the ezTrack wiki

For the original documentation, visit the upstream wiki: https://github.com/DeniseCaiLab/ezTrack/wiki

# Installation and Package Requirements

Installation can be done the same way as described in the original ezTrack wiki, but I recommend using [`uv`](https://docs.astral.sh/uv/). Additionally, this fork requires `ffmpeg`.

1) Download and install `uv` as described here: https://docs.astral.sh/uv/getting-started/installation/

2) Download ezTrack files.

- From GitHub, download and unzip the repository (or clone with `git clone https://github.com/DeniseCaiLab/ezTrack.git`).

3) In the ezTrack directory, use `uv` to create a virtual environment with Python 3.9 and the required packages.

Create the virtual environment:
```bash
uv python install 3.9
uv venv --python 3.9 ez_venv
```

Then, activate it. On Linux or Mac:
```bash
source ez_venv/bin/activate
```

OR on Windows, in PowerShell:
```powershell
ez_venv\Scripts\activate
```

And install all required packages:
```bash
uv pip install -r requirements.txt
```

4) Install FFmpeg (Required for this fork)

This fork uses `ffmpeg` for faster frame extraction. It must be installed and available in your system PATH.

- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```
- **macOS (using Homebrew):**
  ```bash
  brew install ffmpeg
  ```
- **Windows:**
  Download the build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases), extract it, and add the `bin` folder to your System PATH.
  
  *To add to PATH:*
  1. Search for "Environment Variables" in the Start menu and select "Edit the system environment variables".
  2. Click "Environment Variables".
  3. Under "System variables" (or "User variables"), select "Path" and click "Edit".
  4. Click "New" and paste the full path to the `bin` folder inside your extracted ffmpeg folder (e.g., `C:\ffmpeg\bin`).
  5. Click OK to save all dialogs.

  Alternatively, install via `winget` (which handles PATH automatically). Open up the Windows Terminal and run:
  ```powershell
  winget install ffmpeg
  ```

5) To start, activate the environment and start up Jupyter Lab:

```bash
source ez_venv/bin/activate
jupyter lab
```

**Uninstalling ezTrack**

There are no special uninstallation procedures for the environment. Simply delete the `ez_venv` dir and the ezTrack files.

# Location Tracking Module
The location tracking module allows for the analysis of a single animal's location on a frame by frame basis.  In addition to providing the user the with the ability to crop the portion of the video frame in which the animal will be, it also allows the user to specify regions of interest (e.g. left and right sides) and provides tools to quantify the time spent in each region, as well as distance travelled.  
![schematic_lt](../master/Images/LocationTracking_Schematic.png)

### Changes in this fork
This fork introduces several major enhancements to the Location Tracking module, focusing on reproducibility, batch processing efficiency, and flexibility.

#### Programmatic ROI Presets
Instead of manually drawing ROIs, this fork supports defining ROIs programmatically in `Batch_LocationTracking.py`.
- **Open Field (OF):** Automatically define walls, corners, and center based on a `wall_fraction` (rectangular arena)
- **Elevated Plus Maze (EPM):** Automatically define open/closed arms and center based on arm width/height fractions.
- **Morris Water Maze (MWM):** Automatically define an ellipse with four quadrants and a platform (circular arena).
This ensures consistent ROI definitions across all videos in a study.

#### Saving and Reloading Configurations
- **Config Persistence:** The `video_dict` (containing crop coordinates, masks, and parameters) can now be pickled and saved using a custom `DataStub` class.
- **Reloading:** The batch script can load these saved configurations (`video_dict_storefile_*.pickle`), allowing you to apply the same crop/mask settings to multiple videos or resume work easily.
- These config files can also be used for...

#### Enhanced Batch Processing (`Batch_LocationTracking.py`)
- **Parallel Processing:** The batch script now uses `multiprocessing.Pool` to process multiple videos simultaneously.
- **Flexible Configuration:** You can define multiple configuration sets and map them to specific lists of video files.
- **Parameter Overrides:** Selectively override parameters for a batch run without modifying the saved config files.

#### Performance Improvements

- Some functions like reference generation and example frame extraction for thresholding should be noticeably faster. Location tracking should be a bit faster.

More details can be found in the `LocationTracking_Single.ipynb` notebook and in `Batch_LocationTracking.py`.


# Freeze Analysis Module (unchanged)
The freeze analysis module allows the user to automatically score an animal's motion and freezing while in a conditioning chamber.  It was designed with side-view recording in mind, and with the intention of being able to crop the top of a video frame to remove the influence of fiberoptic/miniscope cables.  In the case where no cables are to be used, recording should be capable from above the animal.  
![schematic_fz](../master/Images/FreezeAnalysis_Schematic.png)


# License
This project is licensed under GNU GPLv3.
