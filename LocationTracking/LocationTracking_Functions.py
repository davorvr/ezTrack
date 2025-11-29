"""

LIST OF FUNCTIONS

LoadAndCrop
cropframe
Reference
Locate
TrackLocation
LocationThresh_View
ROI_plot
ROI_Location
Batch_LoadFiles
Batch_Process
PlayVideo
PlayVideo_ext
showtrace
Heatmap
DistanceTool
ScaleDistance

"""





########################################################################################

import os
import sys
import cv2
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import time
import warnings
import functools as fct
import math
import shutil
import subprocess
import multiprocessing as mp
from pathlib import Path
import datetime
from copy import deepcopy
from scipy import ndimage
from tqdm import tqdm
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
from io import BytesIO
from IPython.display import clear_output, Image, display
hv.notebook_extension("bokeh")
warnings.filterwarnings("ignore")

########################################################################################

class DataStub:
    """
    Minimal container with a .data attribute to mimic a stream-like object.
    .data can be any Python object (dict, list, ndarray, etc.).
    """
    __slots__ = ("data",)
    def __init__(self, data=None):
        self.data = data
    def get(self):
        return self.data
    def set(self, value):
        self.data = deepcopy(value)

def copy_video_dict(video_dict: dict):
    vd = {}
    for k,v in video_dict.items():
        if k == "mask" and isinstance(v, dict):
            vd[k] = {}
            for mask_k, mask_v in v.items():
                if mask_k == "stream" and isinstance(mask_v, streams.PolyDraw):
                    vd[k][mask_k] = DataStub(mask_v.data)
                else:
                    vd[k][mask_k] = deepcopy(mask_v)
        elif isinstance(v, streams.BoxEdit) or isinstance(v, streams.PolyDraw):
            vd[k] = DataStub(v.data)
        # elif k == "reference": # better to clear the reference frame and force regeneration
        #     vd[k] = None
        else:
            vd[k] = deepcopy(v)
    return vd

def change_dsmpl_video_dict(video_dict: dict, old_dsmpl: float, new_dsmpl: float):
    vd = {}
    for k,v in video_dict.items():
        if k == "mask" and isinstance(v, dict):
            vd[k] = {}
            for mask_k, mask_v in v.items():
                if mask_k == "stream" and isinstance(mask_v, streams.PolyDraw):
                    vd[k][mask_k] = DataStub(mask_v.data)
                else:
                    vd[k][mask_k] = deepcopy(mask_v)
        elif isinstance(v, streams.BoxEdit) or isinstance(v, streams.PolyDraw):
            vd[k] = DataStub(v.data)
        else:
            vd[k] = deepcopy(v)
    return vd

def process_frame(frame, video_dict,
                  do_gray = True, do_angle = True, do_dsmpl = True, do_crop = True):
    if do_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if video_dict["angle"] and do_angle:
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, video_dict["angle"], 1)
        # Perform the rotation
        frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
    if (video_dict["dsmpl"] < 1) and do_dsmpl:
        frame = cv2.resize(
            frame,
            (
                int(frame.shape[1] * video_dict["dsmpl"]),
                int(frame.shape[0] * video_dict["dsmpl"])
            ),
            cv2.INTER_NEAREST
        )
    if do_crop:
        frame = cropframe(frame, video_dict.get("crop"))
    return frame

def LoadAndCrop(video_dict,cropmethod=None,fstfile=False,accept_p_frames=False,clear_history=False):
    """
    -------------------------------------------------------------------------------------

    Loads video and creates interactive cropping tool (video_dict["crop"] from first frame. In the
    case of batch processing, the first frame of the first video is used. Additionally,
    when batch processing, the same cropping parameters will be appplied to every video.
    Care should therefore be taken that the region of interest is in the same position across
    videos.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        cropmethod:: [str]
            Method of cropping video.  cropmethod takes the following values:
                None : No cropping
                "Box" : Create box selection tool for cropping video

        fstfile:: [bool]
            Dictates whether to use first file in video_dict["FileNames"] to generate
            reference.  True/False

        accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.

    -------------------------------------------------------------------------------------
    Returns:
        image:: [holoviews.Image]
            Holoviews hv.Image displaying first frame

        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection for selection of cropping parameters
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]


    -------------------------------------------------------------------------------------
    Notes:
        - in the case of batch processing, video_dict["file"] is set to first
          video in file
        - prior cropping method HLine has been removed

    """

    #if batch processing, set file to first file to be processed
    video_dict = copy_video_dict(video_dict)
    video_dict["file"] = video_dict["FileNames"][0] if fstfile else video_dict["file"]

    #Upoad file and check that it exists
    # video_dict["fpath"] = os.path.join(os.path.normpath(video_dict["dpath"]), video_dict["file"])
    if video_dict["fpath"].exists():
        print("file: {file}".format(file=str(video_dict["fpath"])))
        cap = cv2.VideoCapture(str(video_dict["fpath"]))
    else:
        raise FileNotFoundError("{file} not found. Check that directory and file names are correct".format(
            file=str(video_dict["fpath"])))

    #Print video information. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frames: {frames}".format(frames=cap_max))
    print("nominal fps: {fps}".format(fps=cap.get(cv2.CAP_PROP_FPS)))
    print("dimensions (h x w): {h},{w}".format(
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

    #check for video p-frames
    if accept_p_frames is False:
        check_p_frames(cap)

    #Set first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict["start"])
    ret, frame = cap.read()
    frame = process_frame(frame, video_dict)
    video_dict["f0"] = frame
    cap.release()

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
    image.opts(
        width=int(frame.shape[1]*video_dict["stretch"]["width"]),
        height=int(frame.shape[0]*video_dict["stretch"]["height"]),
        invert_yaxis=True,
        cmap="gray",
        colorbar=True,
        toolbar="below",
        title="First Frame.  Crop if Desired"
    )

    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod==None:
        image.opts(title="First Frame")
        video_dict["crop"] = None
        return image, None, video_dict

    if cropmethod=="Box":
        poly_dicts = []
        if (
            ("crop" in video_dict) and
            isinstance(video_dict["crop"], DataStub) and
            (video_dict["crop"].data.get("xs", None) or video_dict["crop"].data.get("x", None)) and
            (not clear_history)
        ):
            initial_data = video_dict["crop"].data
            poly_dicts.append({"x": [initial_data["x0"][0], initial_data["x1"][0]],
                               "y": [initial_data["y0"][0], initial_data["y1"][0]]})
        else:
            initial_data = None

        poly = hv.Polygons(poly_dicts)
        box_stream = streams.BoxEdit(source = poly,
                                     num_objects = 1,
                                     data = initial_data)
        poly.opts(fill_alpha = 0.3, active_tools = ["box_edit"])

        return (image*poly), box_stream, video_dict





########################################################################################

def cropframe(frame, crop=None):
    """
    -------------------------------------------------------------------------------------

    Crops passed frame with `crop` specification

    -------------------------------------------------------------------------------------
    Args:
        frame:: [numpy.ndarray]
            2d numpy array
        crop:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to
            cropping tool. `crop.data` contains x and y coordinates of crop
            boundary vertices. Set to None if no cropping supplied.

    -------------------------------------------------------------------------------------
    Returns:
        frame:: [numpy.ndarray]
            2d numpy array

    -------------------------------------------------------------------------------------
    Notes:

    """

    try:
        Xs=[crop.data["x0"][0],crop.data["x1"][0]]
        Ys=[crop.data["y0"][0],crop.data["y1"][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
        return frame[fymin:fymax,fxmin:fxmax]
    except:
        return frame





########################################################################################

# def Reference(video_dict,num_frames=100,
#               altfile=False,fstfile=False,frames=None,segment=None):
#     """
#     -------------------------------------------------------------------------------------
#
#     Generates reference frame by taking median of random subset of frames.  This has the
#     effect of removing animal from frame provided animal is not inactive for >=50% of
#     the video segment.
#
#     -------------------------------------------------------------------------------------
#     Args:
#         video_dict:: [dict]
#             Dictionary with the following keys:
#                 "dpath" : directory containing files [str]
#                 "file" : filename with extension, e.g. "myvideo.wmv" [str]
#                 "start" : frame at which to start. 0-based [int]
#                 "end" : frame at which to end.  set to None if processing
#                         whole video [int]
#                 "region_names" : list of names of regions.  if no regions, set to None
#                 "dsmpl" : proptional degree to which video should be downsampled
#                         by (0-1).
#                 "stretch" : Dictionary used to alter display of frames, with the following keys:
#                         "width" : proportion by which to stretch frame width [float]
#                         "height" : proportion by which to stretch frame height [float]
#                         *Does not influence actual processing, unlike dsmpl.
#                 "reference": Reference image that the current frame is compared to. [numpy.array]
#                 "roi_stream" : Holoviews stream object enabling dynamic selection in response to
#                                selection tool. `poly_stream.data` contains x and y coordinates of roi
#                                vertices. [hv.streams.stream]
#                 "crop" : Enables dynamic box selection of cropping parameters.
#                          Holoviews stream object enabling dynamic selection in response to
#                          `stream.data` contains x and y coordinates of crop boundary vertices.
#                          [hv.streams.BoxEdit]
#                 "mask" : [dict]
#                     Dictionary with the following keys:
#                         "mask" : boolean numpy array identifying regions to exlude
#                                  from analysis.  If no such regions, equal to
#                                  None. [bool numpy array)
#                         "mask_stream" : Holoviews stream object enabling dynamic selection
#                                 in response to selection tool. `mask_stream.data` contains
#                                 x and y coordinates of region vertices. [holoviews polystream]
#                 "scale:: [dict]
#                         Dictionary with the following keys:
#                             "px_distance" : distance between reference points, in pixels [numeric]
#                             "true_distance" : distance between reference points, in desired scale
#                                                (e.g. cm) [numeric]
#                             "true_scale" : string containing name of scale (e.g. "cm") [str]
#                             "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
#                 "ftype" : (only if batch processing)
#                           video file type extension (e.g. "wmv") [str]
#                 "FileNames" : (only if batch processing)
#                               List of filenames of videos in folder to be batch
#                               processed.  [list]
#                 "f0" : (only if batch processing)
#                         first frame of video [numpy array]
#
#         num_frames:: [uint]
#             Number of frames to base reference frame on.
#
#         altfile:: [bool]
#             Specify whether alternative file than video to be processed will be
#             used to generate reference frame. If `altfile=True`, it is expected
#             that `video_dict` contains `altfile` key.
#
#         fstfile:: [bool]
#             Dictates whether to use first file in video_dict["FileNames"] to generate
#             reference.  True/False
#
#         frames:: [np array]
#             User defined selection of frames to use for generating reference
#
#     -------------------------------------------------------------------------------------
#     Returns:
#         reference:: [numpy.array]
#             Reference image. Median of random subset of frames.
#         image:: [holoviews.image]
#             Holoviews Image of reference image.
#
#     -------------------------------------------------------------------------------------
#     Notes:
#         - If `altfile` is specified, it will be used to generate reference.
#
#     """
#     if sum([bool(x) for x in num_frames, frames, segment]) != 1:
#         raise ValueError("Please specify exactly one of `num_frames`, `frames`, and `segment`!")
#
#     #set file to use for reference
#     video_dict["file"] = video_dict["FileNames"][0] if fstfile else video_dict["file"]
#     vname = video_dict.get("altfile","") if altfile else video_dict["file"]
#     fpath = os.path.join(os.path.normpath(video_dict["dpath"]), vname)
#     if os.path.isfile(fpath):
#         cap = cv2.VideoCapture(fpath)
#     else:
#         raise FileNotFoundError("File not found. Check that directory and file names are correct.")
#     cap.set(cv2.CAP_PROP_POS_FRAMES,0)
#
#     #Get video dimensions with any cropping applied
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if (video_dict["dsmpl"] < 1):
#         frame = cv2.resize(
#                     frame,
#                     (
#                         int(frame.shape[1]*video_dict["dsmpl"]),
#                         int(frame.shape[0]*video_dict["dsmpl"])
#                     ),
#                     cv2.INTER_NEAREST)
#     frame = cropframe(
#         frame,
#         video_dict.get("crop")
#     )
#     h,w = frame.shape[0], frame.shape[1]
#     cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap_max = int(video_dict["end"]) if video_dict["end"] is not None else cap_max
#
#     #Collect subset of frames
#     if num_frames:
#         #frames = np.random.randint(video_dict["start"],cap_max,num_frames)
#         frames = np.linspace(start=video_dict["start"], stop=cap_max, num=num_frames)
#     elif frames:
#         num_frames = len(frames) #make sure num_frames equals length of passed list
#     elif segment:
#         frames = tuple(range(segment[0], segment[1]))
#         num_frames = segment[1]-segment[0]
#
#     collection = np.zeros((num_frames,h,w))
#     for (idx,framenum) in enumerate(frames):
#         grabbed = False
#         while grabbed == False:
#             if (not segment) or (idx == 0):
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
#             else:
#                 cap.grab()
#             ret, frame = cap.read()
#             if ret == True:
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 if (video_dict["dsmpl"] < 1):
#                     gray = cv2.resize(
#                         gray,
#                         (
#                             int(gray.shape[1]*video_dict["dsmpl"]),
#                             int(gray.shape[0]*video_dict["dsmpl"])
#                         ),
#                         cv2.INTER_NEAREST)
#                 gray = cropframe(
#                     gray,
#                     video_dict.get("crop")
#                 )
#                 collection[idx,:,:]=gray
#                 grabbed = True
#             elif ret == False:
#                 framenum = np.random.randint(video_dict["start"],cap_max,1)[0]
#                 pass
#     cap.release()
#
#     reference = np.median(collection,axis=0)
#     image = hv.Image((np.arange(reference.shape[1]),
#                       np.arange(reference.shape[0]),
#                       reference)).opts(width=int(reference.shape[1]*video_dict["stretch"]["width"]),
#                                        height=int(reference.shape[0]*video_dict["stretch"]["height"]),
#                                        invert_yaxis=True,
#                                        cmap="gray",
#                                        colorbar=True,
#                                        toolbar="below",
#                                        title="Reference Frame")
#     return reference, image

def Reference_loadimage(image_path):
    ref = cv2.imread(str(image_path))
    return process_frame(ref)

def Reference(video_dict, num_frames=None,
              altfile=False, fstfile=False, frames=None, segment=None):
    """
    -------------------------------------------------------------------------------------

    Generates reference frame by taking median of a subset of frames, helping remove the animal from the reference if it"s not inactive for >=50% of the video.

    -------------------------------------------------------------------------------------
    Args:
        video_dict: [dict] with keys as documented.
        num_frames: [int] Number of frames to base reference frame on.
        altfile: [bool] Use alternative file if present.
        fstfile: [bool] Use the first filename in FileNames.
        frames: [np.array] Specific frames to use.
        segment: [tuple] (start, end), specific video segment.

    -------------------------------------------------------------------------------------
    Returns:
        reference: [numpy.array] Median frame image.
        image: [holoviews.Image] Displayable reference.

    -------------------------------------------------------------------------------------
    Notes:
        Only one argument out of num_frames, frames, and segment should be provided.
    """
    # Argument Validation
    arg_count = sum([num_frames is not None, frames is not None, segment is not None])
    if arg_count != 1:
        raise ValueError("Please specify exactly ONE of num_frames, frames, or segment (not zero or multiple)!")

    # Select file
    vname = (
        video_dict.get("altfile", "")
        if altfile else (video_dict["FileNames"][0] if fstfile else video_dict["file"])
    )
    fpath = video_dict["dpath"].resolve()/vname
    if not fpath.exists():
        raise FileNotFoundError("File not found at %s. Check directory and file names." % str(fpath))

    cap = cv2.VideoCapture(str(fpath))

    # Determine frame range and shape
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(video_dict.get("start", 0))
    end_frame = int(video_dict["end"]) if video_dict.get("end") is not None else total_frames

    # Get image dimensions by reading first frame after applying crop and downsample
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("Could not read frame for shape initialization.")

    frame = process_frame(frame, video_dict)
    h, w = frame.shape[0], frame.shape[1]

    # Generate frame indices
    if frames is not None:
        indices = np.asarray(frames, dtype=int)
        num_frames = len(indices)
    elif segment is not None:
        indices = np.arange(segment[0], segment[1], dtype=int)
        num_frames = len(indices)
    else:  # num_frames given
        indices = np.linspace(start_frame, end_frame - 1, num=num_frames, dtype=int)

    # Preallocate storage
    collection = np.zeros((num_frames, h, w), dtype=np.int16)

    # Ensure indices are sorted and clipped
    indices = np.clip(indices, start_frame, end_frame - 1)
    indices = np.sort(indices)

    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    last_pos = indices[0]
    ret, frame = cap.read()  # Read first frame

    if not ret:
        raise RuntimeError("Could not read initial frame.")

    collection[0, :, :] = process_frame(frame, video_dict)  # your grayscale, resize, crop

    start_time = datetime.datetime.now()
    idx = 1
    for idx, framenum in enumerate(indices[1:], start=1):
        # Calculate how many frames to grab to reach the desired frame
        frames_to_grab = framenum - last_pos - 1
        for _ in range(frames_to_grab):
            cap.grab()  # skip intermediate frames
        # Grab the frame we want to retrieve
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            # fallback in case retrieve failed
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {framenum}")

        last_pos = framenum
        collection[idx, :, :] = process_frame(frame, video_dict)


    cap.release()
    cap_time = datetime.datetime.now()
    print(cap_time-start_time)
    # Compute median frame
    reference = np.median(collection, axis=0).astype(np.int16)
    print(datetime.datetime.now()-cap_time)
    # Prepare holoviews image display
    image = hv.Image(
        (np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference)
    ).opts(
        width=int(reference.shape[1] * video_dict["stretch"]["width"]),
        height=int(reference.shape[0] * video_dict["stretch"]["height"]),
        invert_yaxis=True,
        cmap="gray",
        colorbar=True,
        toolbar="below",
        title="Reference Frame"
    )
    return reference, image




########################################################################################

# def Locate(cap,tracking_params,video_dict,prior=None):
#     """
#     -------------------------------------------------------------------------------------
#
#     Return location of animal in frame, in x/y coordinates.
#
#     -------------------------------------------------------------------------------------
#     Args:
#         cap:: [cv2.VideoCapture]
#             OpenCV VideoCapture class instance for video.
#
#         tracking_params:: [dict]
#             Dictionary with the following keys:
#                 "loc_thresh" : Percentile of difference values below which are set to 0.
#                                After calculating pixel-wise difference between passed
#                                frame and reference frame, these values are tthresholded
#                                to make subsequent defining of center of mass more
#                                reliable. [float between 0-100]
#                 "use_window" : Will window surrounding prior location be
#                                imposed?  Allows changes in area surrounding animal"s
#                                location on previous frame to be more heavily influential
#                                in determining animal"s current location.
#                                After finding pixel-wise difference between passed frame
#                                and reference frame, difference values outside square window
#                                of prior location will be multiplied by (1 - window_weight),
#                                reducing their overall influence. [bool]
#                 "window_size" : If `use_window=True`, the length of one side of square
#                                 window, in pixels. [uint]
#                 "window_weight" : 0-1 scale for window, if used, where 1 is maximal
#                                   weight of window surrounding prior locaiton.
#                                   [float between 0-1]
#                 "method" : "abs", "light", or "dark".  If "abs", absolute difference
#                            between reference and current frame is taken, and thus the
#                            background of the frame doesn"t matter. "light" specifies that
#                            the animal is lighter than the background. "dark" specifies that
#                            the animal is darker than the background.
#                 "rmv_wire" : True/False, indicating whether to use wire removal function.  [bool]
#                 "wire_krn" : size of kernel used for morphological opening to remove wire. [int]
#
#         video_dict:: [dict]
#             Dictionary with the following keys:
#                 "dpath" : directory containing files [str]
#                 "file" : filename with extension, e.g. "myvideo.wmv" [str]
#                 "start" : frame at which to start. 0-based [int]
#                 "end" : frame at which to end.  set to None if processing
#                         whole video [int]
#                 "region_names" : list of names of regions.  if no regions, set to None
#                 "dsmpl" : proptional degree to which video should be downsampled
#                         by (0-1).
#                 "stretch" : Dictionary used to alter display of frames, with the following keys:
#                         "width" : proportion by which to stretch frame width [float]
#                         "height" : proportion by which to stretch frame height [float]
#                         *Does not influence actual processing, unlike dsmpl.
#                 "reference": Reference image that the current frame is compared to. [numpy.array]
#                 "roi_stream" : Holoviews stream object enabling dynamic selection in response to
#                                selection tool. `poly_stream.data` contains x and y coordinates of roi
#                                vertices. [hv.streams.stream]
#                 "crop" : Enables dynamic box selection of cropping parameters.
#                          Holoviews stream object enabling dynamic selection in response to
#                          `stream.data` contains x and y coordinates of crop boundary vertices.
#                          [hv.streams.BoxEdit]
#                 "mask" : [dict]
#                     Dictionary with the following keys:
#                         "mask" : boolean numpy array identifying regions to exlude
#                                  from analysis.  If no such regions, equal to
#                                  None. [bool numpy array)
#                         "mask_stream" : Holoviews stream object enabling dynamic selection
#                                 in response to selection tool. `mask_stream.data` contains
#                                 x and y coordinates of region vertices. [holoviews polystream]
#                 "scale:: [dict]
#                         Dictionary with the following keys:
#                             "px_distance" : distance between reference points, in pixels [numeric]
#                             "true_distance" : distance between reference points, in desired scale
#                                                (e.g. cm) [numeric]
#                             "true_scale" : string containing name of scale (e.g. "cm") [str]
#                             "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
#                 "ftype" : (only if batch processing)
#                           video file type extension (e.g. "wmv") [str]
#                 "FileNames" : (only if batch processing)
#                               List of filenames of videos in folder to be batch
#                               processed.  [list]
#                 "f0" : (only if batch processing)
#                         first frame of video [numpy array]
#
#         prior:: [list]
#             If window is being used, list of length 2 is passed, where first index is
#             prior y position, and second index is prior x position.
#
#     -------------------------------------------------------------------------------------
#     Returns:
#         ret:: [bool]
#             Specifies whether frame is returned in response to cv2.VideoCapture.read.
#
#         dif:: [numpy.array]
#             Pixel-wise difference from prior frame, after thresholding and
#             applying window weight.
#
#         com:: [tuple]
#             Indices of center of mass as tuple in the form: (y,x).
#
#         frame:: [numpy.array]
#             Original video frame after cropping.
#
#     -------------------------------------------------------------------------------------
#     Notes:
#
#     """
#
#     #attempt to load frame
#     ret, frame = cap.read()
#
#     #set window dimensions
#     if prior != None and tracking_params["use_window"]==True:
#         window_size = tracking_params["window_size"]//2
#         ymin,ymax = prior[0]-window_size, prior[0]+window_size
#         xmin,xmax = prior[1]-window_size, prior[1]+window_size
#
#     if ret == True:
#
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         if (video_dict["dsmpl"] < 1):
#             frame = cv2.resize(
#                 frame,
#                 (
#                     int(frame.shape[1]*video_dict["dsmpl"]),
#                     int(frame.shape[0]*video_dict["dsmpl"])
#                 ),
#                 cv2.INTER_NEAREST)
#         frame = cropframe(
#             frame,
#             video_dict.get("crop")
#         )
#
#         #find difference from reference
#         if tracking_params["method"] == "abs":
#             dif = np.absolute(frame-video_dict["reference"])
#         elif tracking_params["method"] == "light":
#             dif = frame-video_dict["reference"]
#         elif tracking_params["method"] == "dark":
#             dif = video_dict["reference"]-frame
#         dif = dif.astype("int16")
#         if "mask" in video_dict.keys():
#             if video_dict["mask"]["mask"] is not None:
#                     dif[video_dict["mask"]["mask"]] = 0
#
#         #apply window
#         weight = 1 - tracking_params["window_weight"]
#         if prior != None and tracking_params["use_window"]==True:
#             dif = dif + (dif.min() * -1) #scale so lowest value is 0
#             dif_weights = np.ones(dif.shape)*weight
#             dif_weights[slice(ymin if ymin>0 else 0, ymax),
#                         slice(xmin if xmin>0 else 0, xmax)]=1
#             dif = dif*dif_weights
#
#         #threshold differences and find center of mass for remaining values
#         dif[dif<np.percentile(dif,tracking_params["loc_thresh"])]=0
#
#         #remove influence of wire
#         if tracking_params["rmv_wire"] == True:
#             ksize = tracking_params["wire_krn"]
#             kernel = np.ones((ksize,ksize),np.uint8)
#             dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
#             krn_violation =  dif_wirermv.sum()==0
#             dif = dif if krn_violation else dif_wirermv
#             if krn_violation:
#                 print("WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {x}".format(
#                     x= int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1-video_dict["start"])))
#
#         com=ndimage.measurements.center_of_mass(dif)
#         return ret, dif, com, frame
#
#     else:
#         return ret, None, None, frame

def Locate(frame, tracking_params, video_dict, prior=None):
    """
    -------------------------------------------------------------------------------------
    Return location of animal in frame, in x/y coordinates, given a decoded frame.

    -------------------------------------------------------------------------------------
    Args:
        frame: [np.ndarray]
            Decoded BGR frame (as returned by cap.read()/cap.retrieve()).

        tracking_params: [dict]
            Keys:
                "loc_thresh" [0–100]
                "use_window" [bool]
                "window_size" [uint]
                "window_weight" [0–1]
                "method" in {"abs","light","dark"}
                "rmv_wire" [bool]
                "wire_krn" [int]

        video_dict: [dict]
            Must contain:
                "reference": grayscale reference (same shape after preprocess)
                "dsmpl": downsample factor in (0,1] (default 1)
                "crop": crop params for cropframe (or None)
                "mask": optional dict with key "mask" (bool array) same shape as reference
                "start": optional, only used previously for messages
                "stretch": not used here but exists in pipeline

        prior: [list-like or tuple] [y, x]
            Prior location for optional window weighting.

    -------------------------------------------------------------------------------------
    Returns:
        ret: [bool] True if frame was valid and processed; False otherwise.
        dif: [np.ndarray or None] processed difference image (float32).
        com: [tuple or None] (y, x) center of mass; (nan, nan) if no mass; None if ret=False.
        frame_proc: [np.ndarray or None] processed grayscale frame (uint8) after resize/crop.
    """
    # Validate input frame
    if frame is None or not isinstance(frame, np.ndarray):
        return False, None, None, None

    # Convert BGR->Gray
    frame = process_frame(frame, video_dict)

    # Reference
    ref = video_dict["reference"]
    if ref.shape != frame.shape:
        raise ValueError(f"Reference shape {ref.shape} does not match frame shape {frame.shape} after preprocess.")

    # Compute difference in float32 for safety and percentile stability
    f32 = frame.astype(np.float32)
    r32 = ref.astype(np.float32)
    method = tracking_params.get("method", "abs")

    if method == "abs":
        dif = np.abs(f32 - r32)
    elif method == "light":
        dif = f32 - r32
        dif[dif < 0] = 0
    elif method == "dark":
        dif = r32 - f32
        dif[dif < 0] = 0
    else:
        dif = np.abs(f32 - r32)

    # Apply mask if present
    mask = video_dict.get("mask", {}).get("mask", None)
    if mask is not None:
        if mask.shape != dif.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match frame shape {dif.shape}.")
        dif[mask] = 0

    # Optional prior-based window weighting
    if tracking_params.get("use_window", False) and prior is not None:
        H, W = dif.shape
        wyx = (int(prior[0]), int(prior[1]))
        window_size = int(tracking_params.get("window_size", 0))
        if window_size > 0:
            half = window_size // 2
            y0 = max(0, wyx[0] - half)
            y1 = min(H, wyx[0] + half)
            x0 = max(0, wyx[1] - half)
            x1 = min(W, wyx[1] + half)
            if y1 > y0 and x1 > x0:
                weight = float(1.0 - tracking_params.get("window_weight", 0.0))
                weight = max(0.0, min(1.0, weight))
                # Shift to non-negative to make multiplicative weighting well-behaved
                if dif.min() < 0:
                    dif = dif - dif.min()
                if weight < 1.0:
                    wmap = np.full(dif.shape, weight, dtype=np.float32)
                    wmap[y0:y1, x0:x1] = 1.0
                    dif = dif * wmap

    # Percentile threshold
    loc_thresh = float(tracking_params.get("loc_thresh", 0.0))
    loc_thresh = max(0.0, min(100.0, loc_thresh))
    if dif.size:
        thr = np.percentile(dif, loc_thresh)
        if thr > 0:
            dif = np.where(dif >= thr, dif, 0.0).astype(np.float32)
        else:
            dif = dif.astype(np.float32)

    # Optional wire removal
    if tracking_params.get("rmv_wire", False):
        ksize = int(tracking_params.get("wire_krn", 0))
        if ksize > 1:
            kernel = np.ones((ksize, ksize), np.uint8)
            maxv = float(dif.max()) if dif.size else 0.0
            if maxv > 0:
                u8 = np.clip(dif * (255.0 / maxv), 0, 255).astype(np.uint8)
                opened = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)
                if opened.sum() > 0:
                    dif = opened.astype(np.float32) * (maxv / 255.0)
                # else: keep original dif

    # Center of mass
    total_mass = dif.sum()
    if total_mass <= 0 or not np.isfinite(total_mass):
        com = (np.nan, np.nan)
    else:
        com = ndimage.center_of_mass(dif)

    return True, dif, com, frame

def Locate2(frame, tracking_params, dsmpl, crop, ref, mask, prior=None):
    """
    -------------------------------------------------------------------------------------
    Return location of animal in frame, in x/y coordinates, given a decoded frame.

    -------------------------------------------------------------------------------------
    Args:
        frame: [np.ndarray]
            Decoded BGR frame (as returned by cap.read()/cap.retrieve()).

        tracking_params: [dict]
            Keys:
                "loc_thresh" [0–100]
                "use_window" [bool]
                "window_size" [uint]
                "window_weight" [0–1]
                "method" in {"abs","light","dark"}
                "rmv_wire" [bool]
                "wire_krn" [int]

        video_dict: [dict]
            Must contain:
                "reference": grayscale reference (same shape after preprocess)
                "dsmpl": downsample factor in (0,1] (default 1)
                "crop": crop params for cropframe (or None)
                "mask": optional dict with key "mask" (bool array) same shape as reference
                "start": optional, only used previously for messages
                "stretch": not used here but exists in pipeline

        prior: [list-like or tuple] [y, x]
            Prior location for optional window weighting.

    -------------------------------------------------------------------------------------
    Returns:
        ret: [bool] True if frame was valid and processed; False otherwise.
        dif: [np.ndarray or None] processed difference image (float32).
        com: [tuple or None] (y, x) center of mass; (nan, nan) if no mass; None if ret=False.
        frame_proc: [np.ndarray or None] processed grayscale frame (uint8) after resize/crop.
    """
    # Validate input frame
    if frame is None or not isinstance(frame, np.ndarray):
        return False, None, None, None

    # Convert BGR->Gray
    gray = process_frame(frame, video_dict)

    # Reference
    # ref = video_dict["reference"]
    # if ref.shape != gray.shape:
    #     raise ValueError(f"Reference shape {ref.shape} does not match frame shape {gray.shape} after preprocess.")

    # Compute difference in float32 for safety and percentile stability
    # f32 = gray.astype(np.float32)
    # r32 = ref.astype(np.float32)
    method = tracking_params.get("method", "abs")

    if method == "abs":
        dif = cv2.absdiff(gray, ref)
    elif method == "light":
        dif = gray.astype(np.int16) - ref.astype(np.int16)
        dif[dif < 0] = 0
    elif method == "dark":
        dif = ref.astype(np.int16) - gray.astype(np.int16)
        dif[dif < 0] = 0
    else:
        dif = cv2.absdiff(gray, ref)

    # Apply mask if present
    # mask = video_dict.get("mask", {}).get("mask", None)
    if mask is not None:
        if mask.shape != dif.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match frame shape {dif.shape}.")
        dif[mask] = 0

    # Optional prior-based window weighting
    if tracking_params.get("use_window", False) and prior is not None:
        H, W = dif.shape
        wyx = (int(prior[0]), int(prior[1]))
        window_size = int(tracking_params.get("window_size", 0))
        if window_size > 0:
            half = window_size // 2
            y0 = max(0, wyx[0] - half)
            y1 = min(H, wyx[0] + half)
            x0 = max(0, wyx[1] - half)
            x1 = min(W, wyx[1] + half)
            if y1 > y0 and x1 > x0:
                weight = float(1.0 - tracking_params.get("window_weight", 0.0))
                weight = max(0.0, min(1.0, weight))
                # Shift to non-negative to make multiplicative weighting well-behaved
                if dif.min() < 0:
                    dif = dif - dif.min()
                if weight < 1.0:
                    wmap = np.full(dif.shape, weight, dtype=np.float32)
                    wmap[y0:y1, x0:x1] = 1.0
                    dif = dif.astype(np.float32) * wmap

    # Percentile threshold
    loc_thresh = float(tracking_params.get("loc_thresh", 0.0))
    loc_thresh = max(0.0, min(100.0, loc_thresh))
    if dif.size:
        thr = np.percentile(dif.astype(np.float32), loc_thresh)
        if thr > 0:
            dif = np.where(dif >= thr, dif, 0.0).astype(np.float32)
        else:
            dif = dif.astype(np.float32)

    # Optional wire removal
    if tracking_params.get("rmv_wire", False):
        ksize = int(tracking_params.get("wire_krn", 0))
        if ksize > 1:
            kernel = np.ones((ksize, ksize), np.uint8)
            maxv = float(dif.max()) if dif.size else 0.0
            if maxv > 0:
                u8 = np.clip(dif * (255.0 / maxv), 0, 255).astype(np.uint8)
                opened = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)
                if opened.sum() > 0:
                    dif = opened.astype(np.float32) * (maxv / 255.0)
                # else: keep original dif

    # Center of mass
    total_mass = dif.sum()
    if total_mass <= 0 or not np.isfinite(total_mass):
        com = (np.nan, np.nan)
    else:
        com = ndimage.center_of_mass(dif)

    return True, dif, com, gray
########################################################################################

# def TrackLocation(video_dict,tracking_params):
#     """
#     -------------------------------------------------------------------------------------
#
#     For each frame in video define location of animal, in x/y coordinates, and distance
#     travelled from previous frame.
#
#     -------------------------------------------------------------------------------------
#     Args:
#         video_dict:: [dict]
#             Dictionary with the following keys:
#                 "dpath" : directory containing files [str]
#                 "file" : filename with extension, e.g. "myvideo.wmv" [str]
#                 "start" : frame at which to start. 0-based [int]
#                 "end" : frame at which to end.  set to None if processing
#                         whole video [int]
#                 "region_names" : list of names of regions.  if no regions, set to None
#                 "dsmpl" : proptional degree to which video should be downsampled
#                         by (0-1).
#                 "stretch" : Dictionary used to alter display of frames, with the following keys:
#                         "width" : proportion by which to stretch frame width [float]
#                         "height" : proportion by which to stretch frame height [float]
#                         *Does not influence actual processing, unlike dsmpl.
#                 "reference": Reference image that the current frame is compared to. [numpy.array]
#                 "roi_stream" : Holoviews stream object enabling dynamic selection in response to
#                                selection tool. `poly_stream.data` contains x and y coordinates of roi
#                                vertices. [hv.streams.stream]
#                 "crop" : Enables dynamic box selection of cropping parameters.
#                          Holoviews stream object enabling dynamic selection in response to
#                          `stream.data` contains x and y coordinates of crop boundary vertices.
#                          [hv.streams.BoxEdit]
#                 "mask" : [dict]
#                     Dictionary with the following keys:
#                         "mask" : boolean numpy array identifying regions to exlude
#                                  from analysis.  If no such regions, equal to
#                                  None. [bool numpy array)
#                         "mask_stream" : Holoviews stream object enabling dynamic selection
#                                 in response to selection tool. `mask_stream.data` contains
#                                 x and y coordinates of region vertices. [holoviews polystream]
#                 "scale:: [dict]
#                         Dictionary with the following keys:
#                             "px_distance" : distance between reference points, in pixels [numeric]
#                             "true_distance" : distance between reference points, in desired scale
#                                                (e.g. cm) [numeric]
#                             "true_scale" : string containing name of scale (e.g. "cm") [str]
#                             "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
#                 "ftype" : (only if batch processing)
#                           video file type extension (e.g. "wmv") [str]
#                 "FileNames" : (only if batch processing)
#                               List of filenames of videos in folder to be batch
#                               processed.  [list]
#                 "f0" : (only if batch processing)
#                         first frame of video [numpy array]
#
#         tracking_params:: [dict]
#             Dictionary with the following keys:
#                 "loc_thresh" : Percentile of difference values below which are set to 0.
#                                After calculating pixel-wise difference between passed
#                                frame and reference frame, these values are tthresholded
#                                to make subsequent defining of center of mass more
#                                reliable. [float between 0-100]
#                 "use_window" : Will window surrounding prior location be
#                                imposed?  Allows changes in area surrounding animal"s
#                                location on previous frame to be more heavily influential
#                                in determining animal"s current location.
#                                After finding pixel-wise difference between passed frame
#                                and reference frame, difference values outside square window
#                                of prior location will be multiplied by (1 - window_weight),
#                                reducing their overall influence. [bool]
#                 "window_size" : If `use_window=True`, the length of one side of square
#                                 window, in pixels. [uint]
#                 "window_weight" : 0-1 scale for window, if used, where 1 is maximal
#                                   weight of window surrounding prior locaiton.
#                                   [float between 0-1]
#                 "method" : "abs", "light", or "dark".  If "abs", absolute difference
#                            between reference and current frame is taken, and thus the
#                            background of the frame doesn"t matter. "light" specifies that
#                            the animal is lighter than the background. "dark" specifies that
#                            the animal is darker than the background.
#                 "rmv_wire" : True/False, indicating whether to use wire removal function.  [bool]
#                 "wire_krn" : size of kernel used for morphological opening to remove wire. [int]
#
#     -------------------------------------------------------------------------------------
#     Returns:
#         df:: [pandas.dataframe]
#             Pandas dataframe with frame by frame x and y locations,
#             distance travelled, as well as video information and parameter values.
#
#     -------------------------------------------------------------------------------------
#     Notes:
#
#     """
#
#     #load video
#     cap = cv2.VideoCapture(video_dict["fpath"])#set file
#     cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict["start"]) #set starting frame
#     cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap_max = int(video_dict["end"]) if video_dict["end"] is not None else cap_max
#
#     #Initialize vector to store motion values in
#     X = np.zeros(cap_max - video_dict["start"])
#     Y = np.zeros(cap_max - video_dict["start"])
#     D = np.zeros(cap_max - video_dict["start"])
#
#     #Loop through frames to detect frame by frame differences
#     time.sleep(.2) #allow printing
#     for f in tqdm(range(len(D))):
#
#         if f>0:
#             yprior = np.around(Y[f-1]).astype(int)
#             xprior = np.around(X[f-1]).astype(int)
#             ret,dif,com,frame = Locate(cap,tracking_params,video_dict,prior=[yprior,xprior])
#         else:
#             ret,dif,com,frame = Locate(cap,tracking_params,video_dict)
#
#         if ret == True:
#             Y[f] = com[0]
#             X[f] = com[1]
#             if f>0:
#                 D[f] = np.sqrt((Y[f]-Y[f-1])**2 + (X[f]-X[f-1])**2)
#         else:
#             #if no frame is detected
#             f = f-1
#             X = X[:f] #Amend length of X vector
#             Y = Y[:f] #Amend length of Y vector
#             D = D[:f] #Amend length of D vector
#             break
#
#     #release video
#     cap.release()
#     time.sleep(.2) #allow printing
#     print("total frames processed: {f}\n".format(f=len(D)))
#
#     #create pandas dataframe
#     df = pd.DataFrame(
#     {"File" : video_dict["file"],
#      "Location_Thresh": np.ones(len(D))*tracking_params["loc_thresh"],
#      "Use_Window": str(tracking_params["use_window"]),
#      "Window_Weight": np.ones(len(D))*tracking_params["window_weight"],
#      "Window_Size": np.ones(len(D))*tracking_params["window_size"],
#      "Start_Frame": np.ones(len(D))*video_dict["start"],
#      "Frame": np.arange(len(D)),
#      "X": X,
#      "Y": Y,
#      "Distance_px": D
#     })
#
#     #add region of interest info
#     df = ROI_Location(video_dict, df)
#     if video_dict["region_names"] is not None:
#         print("Defining transitions...")
#         df["ROI_location"] = ROI_linearize(df[video_dict["region_names"]])
#         df["ROI_transition"] = ROI_transitions(df["ROI_location"])
#
#     #update scale, if known
#     df = ScaleDistance(video_dict, df=df, column="Distance_px")
#
#     return df

def TrackLocation(video_dict, tracking_params):
    cap = cv2.VideoCapture(str(video_dict["fpath"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_dict["start"]))
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_max = int(video_dict["end"]) if video_dict["end"] is not None else cap_max

    total = max(0, cap_max - int(video_dict["start"]))
    X = np.zeros(total, dtype=np.float32)
    Y = np.zeros(total, dtype=np.float32)
    D = np.zeros(total, dtype=np.float32)

    processed = 0
    if tracking_params["progress_bar"]:
        iterator = tqdm(range(total))
    else:
        iterator = range(total)
    for f in iterator:
        # Sequential read: grab() then retrieve() to avoid blocking decode until needed
        if not cap.grab():
            break
        ret, frame_bgr = cap.retrieve()
        if not ret or frame_bgr is None:
            # fallback: try read() once
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

        # prior for windowing
        prior = None
        if f > 0:
            prior = [int(np.around(Y[f-1])), int(np.around(X[f-1]))]

        ret_loc, dif, com, frame_gray = Locate(frame_bgr, tracking_params, video_dict, prior=prior)
        if not ret_loc or com is None:
            break

        Y[f] = com[0]
        X[f] = com[1]
        if f > 0:
            dy = Y[f] - Y[f-1]
            dx = X[f] - X[f-1]
            D[f] = np.sqrt(dy*dy + dx*dx)

        processed += 1

    # Trim arrays to processed length
    if processed < total:
        X = X[:processed]; Y = Y[:processed]; D = D[:processed]

    cap.release()
    print(f"total frames processed: {len(D)}\n")

    df = pd.DataFrame({
        "File": video_dict["file"],
        "Location_Thresh": np.full(len(D), tracking_params["loc_thresh"], dtype=np.float32),
        "Use_Window": str(tracking_params["use_window"]),
        "Window_Weight": np.full(len(D), tracking_params["window_weight"], dtype=np.float32),
        "Window_Size": np.full(len(D), tracking_params["window_size"], dtype=np.int32),
        "Start_Frame": np.full(len(D), int(video_dict["start"]), dtype=np.int32),
        "Frame": np.arange(len(D), dtype=np.int32),
        "X": X, "Y": Y, "Distance_px": D
    })

    df = ROI_Location(video_dict, df)
    if video_dict["region_names"] is not None:
        print("Defining transitions...")
        df["ROI_location"] = ROI_linearize(df[video_dict["region_names"]])
        df["ROI_transition"] = ROI_transitions(df["ROI_location"])
    df = ScaleDistance(video_dict, df=df, column="Distance_px")
    return df

# def TrackLocation_worker(args):
#     part_path, seg_idx, video_dict, tracking_params = args
#     cap = cv2.VideoCapture(str(part_path), cv2.CAP_FFMPEG)
#     # cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     X = np.zeros(total, dtype=np.float32)
#     Y = np.zeros(total, dtype=np.float32)
#     D = np.zeros(total, dtype=np.float32)
#     ref = video_dict["reference"].astype(np.uint8)
#     mask = video_dict["mask"]["mask"].astype(np.uint8)
#
#     processed = 0
#     t_start = datetime.datetime.now()
#     for f in range(total):
#         # Sequential read: grab() then retrieve() to avoid blocking decode until needed
#         if not cap.grab():
#             break
#         ret, frame_bgr = cap.retrieve()
#         if not ret or frame_bgr is None:
#             # fallback: try read() once
#             ret, frame_bgr = cap.read()
#             if not ret or frame_bgr is None:
#                 break
#
#         # prior for windowing
#         prior = None
#         if f > 0:
#             prior = [int(np.around(Y[f-1])), int(np.around(X[f-1]))]
#
#         # ret_loc, dif, com, frame_gray = Locate2(frame_bgr, tracking_params, video_dict["dsmpl"], video_dict["crop"], ref, mask, prior=prior)
#         ret_loc, dif, com, frame_gray = Locate(frame_bgr, tracking_params, video_dict, prior=prior)
#         if not ret_loc or com is None:
#             break
#
#         Y[f] = com[0]
#         X[f] = com[1]
#         if f > 0:
#             dy = Y[f] - Y[f-1]
#             dx = X[f] - X[f-1]
#             D[f] = np.sqrt(dy*dy + dx*dx)
#         processed += 1
#
#     # Trim arrays to processed length
#     if processed < total:
#         X = X[:processed]; Y = Y[:processed]; D = D[:processed]
#
#     cap.release()
#     # print(f"total frames processed: {len(D)}\n")
#
#     df = pd.DataFrame({
#         "File": video_dict["file"],
#         "Location_Thresh": np.full(len(D), tracking_params["loc_thresh"], dtype=np.float32),
#         "Use_Window": str(tracking_params["use_window"]),
#         "Window_Weight": np.full(len(D), tracking_params["window_weight"], dtype=np.float32),
#         "Window_Size": np.full(len(D), tracking_params["window_size"], dtype=np.int32),
#         "Start_Frame": np.full(len(D), int(video_dict["start"]), dtype=np.int32),
#         "Frame": np.arange(len(D), dtype=np.int32),
#         "X": X, "Y": Y, "Distance_px": D
#     })
#     return seg_idx, df, datetime.datetime.now()-t_start
#
# def ffprobe_duration(path):
#     cmd = [
#         "ffprobe", "-v", "error", "-show_entries", "format=duration",
#         "-of", "default=noprint_wrappers=1:nokey=1", str(path)
#     ]
#     out = subprocess.check_output(cmd).decode().strip()
#     return float(out)
#
# def split_video_segments(src_path, out_dir: Path, n_segments, vid_start_s):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     if vid_start_s < 0:
#         vid_start_s += ffprobe_duration(src_path)
#     dur = ffprobe_duration(src_path)-vid_start_s
#     seg_dur = dur / n_segments
#     parts = []
#     for i in range(n_segments):
#         start = vid_start_s + i * seg_dur
#         # Last segment can run to end
#         out_path = out_dir/f"part_{i:02d}.mp4"
#         # -c copy is fast; -reset_timestamps 1 makes outputs start at 0
#         cmd = [
#             "ffmpeg",
#             "-hide_banner",
#             "-loglevel", "error",
#             "-y",
#             "-ss", f"{start:.6f}",
#             "-i", str(src_path),
#             "-t", f"{seg_dur:.6f}" if i < n_segments-1 else f"{dur - start:.6f}",
#             "-c", "copy",
#             "-avoid_negative_ts", "1",
#             "-reset_timestamps", "1",
#             str(out_path)
#         ]
#         subprocess.run(cmd, check=True)
#         parts.append((out_path, start))
#     return parts
#
# def TrackLocation_parallel(video_dict, tracking_params, n_workers, tmp_dir="_parts"):
#     tmp_dir = Path(tmp_dir)
#     if not tmp_dir.is_absolute():
#         tmp_dir = video_dict["output_path"]/tmp_dir
#     tmp_dir.mkdir(parents=True, exist_ok=True)
#     # Split with ffmpeg (returns [(file_path, start_seconds), ...] in order)
#     parts = split_video_segments(video_dict["fpath"], tmp_dir, n_workers, video_dict["start_s"])
#
#     # Map (segment_path, segment_index)
#     args = [(parts[i][0], i, copy_video_dict(video_dict), tracking_params) for i in range(len(parts))]
#
#     # results = []
#     with mp.Pool(processes=n_workers) as pool:
#         # results = pool.map(TrackLocation_worker, args)
#         results = pool.map(TrackLocation_worker, args)
#     print("Pool processed!")
#     # results = [process_segment(args[0])]
#
#     # Sort by segment index and concatenate
#     results.sort(key=lambda t: t[0])   # 0,1,2...
#     dfs = [t[1] for t in results]
#     t_per_segment = [t[2] for t in results]
#     print(t_per_segment)
#     df = pd.concat(dfs, ignore_index=True)
#     df = df.reset_index(drop=True)
#     df["Frame"] = df.index.astype(np.int32)
#
#     # ROI/scaling
#     df = ROI_Location(video_dict, df)
#     if video_dict["region_names"] is not None:
#         print("Defining transitions…")
#         df["ROI_location"] = ROI_linearize(df[video_dict["region_names"]])
#         df["ROI_transition"] = ROI_transitions(df["ROI_location"])
#     df = ScaleDistance(video_dict, df=df, column="Distance_px")
#
#     # Remove temp directory and split files
#     shutil.rmtree(tmp_dir, ignore_errors=True)
#
#     return df


########################################################################################

# def LocationThresh_View(video_dict,tracking_params,examples=4):
#     """
#     -------------------------------------------------------------------------------------
#
#     Display example tracking with selected parameters for a random subset of frames.
#     NOTE that because individual frames are analyzed independently, weighting
#     based upon prior location is not implemented.
#
#     -------------------------------------------------------------------------------------
#     Args:
#
#         video_dict:: [dict]
#             Dictionary with the following keys:
#                 "dpath" : directory containing files [str]
#                 "file" : filename with extension, e.g. "myvideo.wmv" [str]
#                 "start" : frame at which to start. 0-based [int]
#                 "end" : frame at which to end.  set to None if processing
#                         whole video [int]
#                 "region_names" : list of names of regions.  if no regions, set to None
#                 "dsmpl" : proptional degree to which video should be downsampled
#                         by (0-1).
#                 "stretch" : Dictionary used to alter display of frames, with the following keys:
#                         "width" : proportion by which to stretch frame width [float]
#                         "height" : proportion by which to stretch frame height [float]
#                         *Does not influence actual processing, unlike dsmpl.
#                 "reference": Reference image that the current frame is compared to. [numpy.array]
#                 "roi_stream" : Holoviews stream object enabling dynamic selection in response to
#                                selection tool. `poly_stream.data` contains x and y coordinates of roi
#                                vertices. [hv.streams.stream]
#                 "crop" : Enables dynamic box selection of cropping parameters.
#                          Holoviews stream object enabling dynamic selection in response to
#                          `stream.data` contains x and y coordinates of crop boundary vertices.
#                          [hv.streams.BoxEdit]
#                 "mask" : [dict]
#                     Dictionary with the following keys:
#                         "mask" : boolean numpy array identifying regions to exlude
#                                  from analysis.  If no such regions, equal to
#                                  None. [bool numpy array)
#                         "mask_stream" : Holoviews stream object enabling dynamic selection
#                                 in response to selection tool. `mask_stream.data` contains
#                                 x and y coordinates of region vertices. [holoviews polystream]
#                 "scale:: [dict]
#                         Dictionary with the following keys:
#                             "px_distance" : distance between reference points, in pixels [numeric]
#                             "true_distance" : distance between reference points, in desired scale
#                                                (e.g. cm) [numeric]
#                             "true_scale" : string containing name of scale (e.g. "cm") [str]
#                             "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
#                 "ftype" : (only if batch processing)
#                           video file type extension (e.g. "wmv") [str]
#                 "FileNames" : (only if batch processing)
#                               List of filenames of videos in folder to be batch
#                               processed.  [list]
#                 "f0" : (only if batch processing)
#                         first frame of video [numpy array]
#
#         tracking_params:: [dict]
#             Dictionary with the following keys:
#                 "loc_thresh" : Percentile of difference values below which are set to 0.
#                                After calculating pixel-wise difference between passed
#                                frame and reference frame, these values are tthresholded
#                                to make subsequent defining of center of mass more
#                                reliable. [float between 0-100]
#                 "use_window" : Will window surrounding prior location be
#                                imposed?  Allows changes in area surrounding animal"s
#                                location on previous frame to be more heavily influential
#                                in determining animal"s current location.
#                                After finding pixel-wise difference between passed frame
#                                and reference frame, difference values outside square window
#                                of prior location will be multiplied by (1 - window_weight),
#                                reducing their overall influence. [bool]
#                 "window_size" : If `use_window=True`, the length of one side of square
#                                 window, in pixels. [uint]
#                 "window_weight" : 0-1 scale for window, if used, where 1 is maximal
#                                   weight of window surrounding prior locaiton.
#                                   [float between 0-1]
#                 "method" : "abs", "light", or "dark".  If "abs", absolute difference
#                            between reference and current frame is taken, and thus the
#                            background of the frame doesn"t matter. "light" specifies that
#                            the animal is lighter than the background. "dark" specifies that
#                            the animal is darker than the background.
#                 "rmv_wire" : True/False, indicating whether to use wire removal function.  [bool]
#                 "wire_krn" : size of kernel used for morphological opening to remove wire. [int]
#
#         examples:: [uint]
#             The number of frames for location tracking to be tested on.
#
#
#     -------------------------------------------------------------------------------------
#     Returns:
#         df:: [holoviews.Layout]
#             Returns Holoviews Layout with original images on left and heat plots with
#             animal"s estimated position marked on right.
#
#     -------------------------------------------------------------------------------------
#     Notes:
#         - if `stretch` values are modified, this will only influence display and not
#           calculation
#
#     """
#
#     #load video
#     cap = cv2.VideoCapture(video_dict["fpath"])
#     cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap_max = int(video_dict["end"]) if video_dict["end"] is not None else cap_max
#
#     #examine random frames
#     images = []
#     for example in range (examples):
#
#         #analyze frame
#         ret = False
#         while ret is False:
#             frm=np.random.randint(video_dict["start"],cap_max) #select random frame
#             cap.set(cv2.CAP_PROP_POS_FRAMES,frm) #sets frame to be next to be grabbed
#             ret,dif,com,frame = Locate(cap, tracking_params, video_dict)
#
#         #plot original frame
#         image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
#         image_orig.opts(
#             width=int(video_dict["reference"].shape[1]*video_dict["stretch"]["width"]),
#             height=int(video_dict["reference"].shape[0]*video_dict["stretch"]["height"]),
#             invert_yaxis=True,cmap="gray",toolbar="below",
#             title="Frame: " + str(frm))
#         orig_overlay = image_orig * hv.Points(([com[1]],[com[0]])).opts(
#             color="red",size=20,marker="+",line_width=3)
#
#         #plot heatmap
#         dif = dif*(255//dif.max())
#         image_heat = hv.Image((
#             np.arange(dif.shape[1]),
#             np.arange(dif.shape[0]),
#             dif))
#         image_heat.opts(
#             width=int(dif.shape[1]*video_dict["stretch"]["width"]),
#             height=int(dif.shape[0]*video_dict["stretch"]["height"]),
#             invert_yaxis=True,cmap="jet",toolbar="below",
#             title="Frame: " + str(frm - video_dict["start"]))
#         heat_overlay = image_heat * hv.Points(([com[1]],[com[0]])).opts(
#             color="red",size=20,marker="+",line_width=3)
#
#         images.extend([orig_overlay,heat_overlay])
#
#     cap.release()
#     layout = hv.Layout(images)
#     return layout

# def LocationThresh_View(video_dict, tracking_params, examples=4):
#     """
#     Display example tracking with selected parameters for a random subset of frames.
#     NOTE: individual frames are analyzed independently; no prior-location weighting.
#     Returns a Holoviews Layout with original images and heatmaps.
#     """
#     # Open video
#     cap = cv2.VideoCapture(video_dict["fpath"])
#     cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap_max = int(video_dict["end"]) if video_dict["end"] is not None else cap_max
#     start = int(video_dict.get("start", 0))
#     end = cap_max
#
#     images = []
#     for _ in range(examples):
#         # Try until a valid frame is obtained
#         got = False
#         attempts = 0
#         while not got and attempts < 20:
#             attempts += 1
#             frm = np.random.randint(start, end)  # select random frame
#             # Seek to target frame and read
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(frm))
#             ret, frame_bgr = cap.read()
#             if not ret or frame_bgr is None:
#                 continue
#
#             # Analyze using new Locate (no prior weighting here)
#             ret_loc, dif, com, frame = Locate(frame_bgr, tracking_params, video_dict, prior=None)
#             if not ret_loc or dif is None or com is None:
#                 continue
#
#             got = True
#
#         if not got:
#             # If repeatedly failed to read/locate, skip this example
#             continue
#
#         # Plot original frame
#         image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
#         image_orig.opts(
#             width=int(video_dict["reference"].shape[1] * video_dict["stretch"]["width"]),
#             height=int(video_dict["reference"].shape[0] * video_dict["stretch"]["height"]),
#             invert_yaxis=True, cmap="gray", toolbar="below",
#             title=f"Frame: {frm}"
#         )
#         orig_overlay = image_orig * hv.Points(([com[1]], [com[0]])).opts(
#             color="red", size=20, marker="+", line_width=3
#         )
#
#         # Plot heatmap (safe normalization)
#         dmax = int(dif.max()) if dif.size else 0
#         if dmax > 0:
#             # Scale to 0..255 preserving type (dif is int16); cast to uint8 for display
#             dif_vis = ((dif.astype(np.int32) * 255) // dmax).astype(np.uint8)
#         else:
#             dif_vis = np.zeros_like(dif, dtype=np.uint8)
#
#         image_heat = hv.Image(
#             (np.arange(dif_vis.shape[1]), np.arange(dif_vis.shape[0]), dif_vis)
#         ).opts(
#             width=int(dif_vis.shape[1] * video_dict["stretch"]["width"]),
#             height=int(dif_vis.shape[0] * video_dict["stretch"]["height"]),
#             invert_yaxis=True, cmap="jet", toolbar="below",
#             title=f"Frame: {frm - start}"
#         )
#         heat_overlay = image_heat * hv.Points(([com[1]], [com[0]])).opts(
#             color="red", size=20, marker="+", line_width=3
#         )
#
#         images.extend([orig_overlay, heat_overlay])
#
#     cap.release()
#     layout = hv.Layout(images)
#     return layout

def LocationThresh_View(video_dict, tracking_params, examples=4, tmp_dir="_frames"):
    """
    Display example tracking with selected parameters for a random subset of frames.
    NOTE: individual frames are analyzed independently; no prior-location weighting.
    Returns a Holoviews Layout with original images and heatmaps.
    """
    tmp_dir = Path(tmp_dir)
    if not tmp_dir.is_absolute():
        tmp_dir = video_dict["output_path"]/tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    framerate = video_dict["fps"]
    frames = np.sort(np.random.choice(np.arange(video_dict["start"], video_dict["end"]), size=examples, replace=False))
    timestamps = { n: n/video_dict["fps"] for n in frames }
    frames_cmd = [f"eq(n,{i})" for i in np.nditer(frames)]
    frames_cmd_str = "select='" + "+".join(frames_cmd) + "'"
    images = []

    t_frame_extract = datetime.timedelta(0)
    t_locate = datetime.timedelta(0)
    t_draw_image = datetime.timedelta(0)

    out_img = tmp_dir/f"out_frame_%d.png"
    # t_start = datetime.datetime.now()
    # cmd = [
    #     "ffmpeg", "-v", "error",
    #     "-i", str(video_dict["fpath"]),
    #     "-vf", frames_cmd_str,
    #     "-vsync", "0",
    #     str(out_img)
    # ]
    # subprocess.run(cmd, check=True)
    # t_frame_extract += (datetime.datetime.now()-t_start)
    # print(f"t_frame_extract: {t_frame_extract}")

    t_start = datetime.datetime.now()
    cmds = []
    for n, ts in timestamps.items():
        cmds.append([
            "ffmpeg", "-y",
            "-v", "error",
            "-ss", f"{ts:.4f}",
            "-i", str(video_dict["fpath"]),
            "-frames:v", "1",
            str(out_img).replace(f"%d", str(n))
        ])
    processes = [ subprocess.Popen(cmd) for cmd in cmds ]
    for p in processes:
        p.wait()
    t_frame_extract += (datetime.datetime.now()-t_start)


    for n in np.nditer(frames):
        frame = cv2.imread(str(out_img).replace(f"%d", str(n)))
        t_frame_extract += (datetime.datetime.now()-t_start)

        t_start = datetime.datetime.now()
        ret_loc, dif, com, frame = Locate(frame, tracking_params, video_dict, prior=None)
        t_locate += (datetime.datetime.now()-t_start)
        if not ret_loc or dif is None or com is None:
            continue

        # Plot original frame
        t_start = datetime.datetime.now()
        image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
        # baseframe = cropframe(video_dict["f0"], video_dict["crop"])
        # baseframe = process_frame(video_dict["f0"], video_dict)
        baseframe = video_dict["f0"]
        image_orig.opts(
            width=int(baseframe.shape[1] * video_dict["stretch"]["width"]),
            height=int(baseframe.shape[0] * video_dict["stretch"]["height"]),
            invert_yaxis=True, cmap="gray", toolbar="below",
            title=f"Frame: {n}"
        )
        orig_overlay = image_orig * hv.Points(([com[1]], [com[0]])).opts(
            color="red", size=20, marker="+", line_width=3
        )

        # Plot heatmap (safe normalization)
        dmax = int(dif.max()) if dif.size else 0
        if dmax > 0:
            # Scale to 0..255 preserving type (dif is int16); cast to uint8 for display
            dif_vis = ((dif.astype(np.int32) * 255) // dmax).astype(np.uint8)
        else:
            dif_vis = np.zeros_like(dif, dtype=np.uint8)

        image_heat = hv.Image(
            (np.arange(dif_vis.shape[1]), np.arange(dif_vis.shape[0]), dif_vis)
        ).opts(
            width=int(dif_vis.shape[1] * video_dict["stretch"]["width"]),
            height=int(dif_vis.shape[0] * video_dict["stretch"]["height"]),
            invert_yaxis=True, cmap="jet", toolbar="below",
            title=f"Frame: {n}"
        )
        heat_overlay = image_heat * hv.Points(([com[1]], [com[0]])).opts(
            color="red", size=20, marker="+", line_width=3
        )

        images.extend([orig_overlay, heat_overlay])
        t_draw_image += (datetime.datetime.now()-t_start)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    layout = hv.Layout(images)
    print(f"t_frame_extract: {t_frame_extract}")
    print(f"t_locate: {t_locate}")
    print(f"t_draw_image: {t_draw_image}")
    return layout



########################################################################################

def ROI_plot(video_dict, arena_only=False, clear_history=False):
    """
    -------------------------------------------------------------------------------------

    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn. If `arena_only=True`, one polygon can be drawn.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]


    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define regions of interest.

        poly_stream:: [hv.streams.stream]
            Holoviews stream object enabling dynamic selection in response to
            selection tool. `poly_stream.data` contains x and y coordinates of roi
            vertices.

    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation

    """

    #get number of objects to be drawn
    if arena_only:
        nobjects = 1
        title = "Draw regions: arena"
    else:
        nobjects = len(video_dict["region_names"]) if video_dict["region_names"] else 0
        title = "Draw Regions: "+", ".join(video_dict["region_names"])

    #baseframe = cropframe(video_dict["f0"], video_dict["crop"])
    # baseframe = process_frame(video_dict["f0"], video_dict)
    baseframe = video_dict["f0"]
    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(baseframe.shape[1]),
        np.arange(baseframe.shape[0]),
        baseframe))
    image.opts(
        width=int(baseframe.shape[1]*video_dict["stretch"]["width"]),
        height=int(baseframe.shape[0]*video_dict["stretch"]["height"]),
        invert_yaxis=True,cmap="gray", colorbar=True,toolbar="below",
        title="No Regions to Draw" if nobjects == 0 else title)

    poly_dicts = []
    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    if (
        ("roi_stream" in video_dict) and
        (isinstance(video_dict["roi_stream"], streams.PolyDraw) or isinstance(video_dict["roi_stream"], DataStub)) and
        (video_dict["roi_stream"].data.get("xs", None) or video_dict["roi_stream"].data.get("x", None)) and
        (not clear_history)
       ):
        initial_data = video_dict["roi_stream"].data
        for i in range(len(initial_data["xs"])):
            poly_dicts.append({
                "x": initial_data["xs"][i],
                "y": initial_data["ys"][i]})

    else:
        initial_data = None

    poly = hv.Polygons(poly_dicts)
    poly_stream = streams.PolyDraw(source=poly,
                                   drag=True,
                                   num_objects=nobjects,
                                   show_vertices=True,
                                   data=initial_data)
    poly.opts(fill_alpha=0.3, active_tools=["poly_draw"])


    def centers(data):
        try:
            x_ls, y_ls = data["xs"], data["ys"]
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        if arena_only:
            rois = ["arena"]
        else:
            rois = video_dict["region_names"][:len(xs)]
        return hv.Labels((xs, ys, rois))

    if nobjects > 0:
        dmap = hv.DynamicMap(centers, streams=[poly_stream])
        return (image * poly * dmap), poly_stream
    else:
        return (image),None





########################################################################################

def ROI_Location(video_dict, location):
    """
    -------------------------------------------------------------------------------------

    For each frame, determine which regions of interest the animal is in.  For each
    region of interest, boolean array is added to `location` dataframe passed, with
    column name being the region name.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
            Must contain column names "X" and "Y".

    -------------------------------------------------------------------------------------
    Returns:
        location:: [pandas.dataframe]
            For each region of interest, boolean array is added to `location` dataframe
            passed, with column name being the region name. Additionally, under column
            `ROI_coordinates`, coordinates of vertices of each region of interest are
            printed. This takes the form of a dictionary of x and y coordinates, e.g.:
                "xs" : [[region 1 x coords], [region 2 x coords]],
                "ys" : [[region 1 y coords], [region 2 y coords]]

    -------------------------------------------------------------------------------------
    Notes:

    """

    if video_dict["region_names"] == None:
        return location

    #Create ROI Masks
    ROI_masks = {}
    for poly in range(len(video_dict["roi_stream"].data["xs"])):
        x = np.array(video_dict["roi_stream"].data["xs"][poly]) #x coordinates
        y = np.array(video_dict["roi_stream"].data["ys"][poly]) #y coordinates
        xy = np.column_stack((x,y)).astype("uint64") #xy coordinate pairs
        #baseframe = cropframe(video_dict["f0"], video_dict["crop"])
        #baseframe = process_frame(video_dict["f0"], video_dict)
        baseframe = video_dict["f0"]
        mask = np.zeros(baseframe.shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon
        ROI_masks[video_dict["region_names"][poly]] = mask==255 #save to ROI masks as boolean

    #Create arrays to store whether animal is within given ROI
    ROI_location = {}
    for mask in ROI_masks:
        ROI_location[mask]=np.full(len(location["Frame"]),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in location["Frame"]:
        y,x = location["Y"][f], location["X"][f]
        for mask in ROI_masks:
            ROI_location[mask][f] = ROI_masks[mask][int(y),int(x)]

    #Add data to location data frame
    for x in ROI_location:
        location[x]=ROI_location[x]

    #Add ROI coordinates
    location["ROI_coordinates"]=str(video_dict["roi_stream"].data)

    return location





########################################################################################

def ROI_linearize(rois, null_name = "non_roi"):

    """
    -------------------------------------------------------------------------------------

    Creates array defining ROI as string for each frame

    -------------------------------------------------------------------------------------
    Args:
        rois:: [pd.DataFrame]
            Pandas dataframe where each column corresponds to an ROI, with boolean values
            defining if animal is in said roi.
        null_name:: [string]
            Name used when animals is not in any defined roi.

    -------------------------------------------------------------------------------------
    Returns:
        rois["ROI_location"]:: [pd.Series]
            pd.Series defining ROI as string for each frame

    -------------------------------------------------------------------------------------
    Notes:

    """
    region_names = rois.columns.values
    rois["ROI_location"] = null_name
    for region in region_names:
        rois["ROI_location"][rois[region]] = rois["ROI_location"][rois[region]].apply(
            lambda x: "_".join([x, region]) if x!=null_name else region
        )
    return rois["ROI_location"]






########################################################################################

def ROI_transitions(regions, include_first=False):
    """
    -------------------------------------------------------------------------------------

    Creates boolean array defining where transitions between each ROI occur.

    -------------------------------------------------------------------------------------
    Args:
        regions:: [Pandas Series]
            Pandas Series defining ROI as string for each frame
        include_first:: [string]
            Whether to count first frame as transition

    -------------------------------------------------------------------------------------
    Returns:
        transitions:: [Boolean array]
            pd.Series defining where transitions between ROIs occur.

    -------------------------------------------------------------------------------------
    Notes:

    """
    regions_offset = np.append(regions[0], regions[0:-1])
    transitions = regions!=regions_offset
    if include_first:
        transitions[0] = True
    return transitions





########################################################################################

def Summarize_Location(location, video_dict, bin_dict=None):
    """
    -------------------------------------------------------------------------------------

    Generates summary of distance travelled and proportional time spent in each region
    of interest according to user defined time bins.  If bins are not provided
    (`bin_dict=None`), average of entire video segment will be provided.

    -------------------------------------------------------------------------------------
    Args:
        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
            Additionally, for each region of interest, boolean array indicating whether
            animal is in the given region for each frame.

        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        bin_dict:: [dict]
            Dictionary specifying bins.  Dictionary keys should be names of the bins.
            Dictionary value for each bin should be a tuple, with the start and end of
            the bin, in seconds, relative to the start of the analysis period
            (i.e. if start frame is 100, it will be relative to that). If no bins are to
            be specified, set bin_dict = None.
            example: bin_dict = {1:(0,100), 2:(100,200)}


    -------------------------------------------------------------------------------------
    Returns:
        bins:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each
            region of interest according to user defined time bins, as well as video
            information and parameter values. If no region names are supplied
            (`region_names=None`), only distance travelled will be included.

    -------------------------------------------------------------------------------------
    Notes:

    """

    #define bins
    avg_dict = {"all": (location["Frame"].min(), location["Frame"].max())}
    bin_dict = bin_dict if bin_dict is not None else avg_dict

    #get summary info
    bins = (pd.Series(bin_dict).rename("range(f)")
            .reset_index().rename(columns=dict(index="bin")))
    bins["Distance_px"] = bins["range(f)"].apply(
        lambda r: location[location["Frame"].between(*r)]["Distance_px"].sum())
    if video_dict["region_names"] is not None:
        bins_reg = bins["range(f)"].apply(
            lambda r: location[location["Frame"].between(*r)][video_dict["region_names"]].mean())
        bins = bins.join(bins_reg)
        drp_cols = ["Distance_px", "Frame", "X", "Y"] + video_dict["region_names"]
    else:
        drp_cols = ["Distance_px", "Frame", "X", "Y"]
    bins = pd.merge(
        location.drop(drp_cols, axis="columns"),
        bins,
        left_index=True,
        right_index=True)

    #scale distance
    bins = ScaleDistance(video_dict,df=bins,column="Distance_px")

    return bins





########################################################################################

def Batch_LoadFiles(video_dict):
    """
    -------------------------------------------------------------------------------------

    Populates list of files in directory (`dpath`) that are of the specified file type
    (`ftype`).  List is held in `video_dict["FileNames"]`.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]


    -------------------------------------------------------------------------------------
    Returns:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

    -------------------------------------------------------------------------------------
    Notes:

    """

    #Get list of video files of designated type
    if video_dict["dpath"].exists():
        video_dict["FileNames"] = video_dict["dpath"].glob("*." + video_dict["ftype"], case_sensitive=False)
        return video_dict
    else:
        raise FileNotFoundError("{path} not found. Check that directory is correct".format(
            path=str(video_dict["dpath"])))





########################################################################################

def Batch_Process(video_dict,tracking_params,bin_dict,accept_p_frames=False):
    """
    -------------------------------------------------------------------------------------

    Run LocationTracking on folder of videos of specified filetype.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        tracking_params:: [dict]
            Dictionary with the following keys:
                "loc_thresh" : Percentile of difference values below which are set to 0.
                               After calculating pixel-wise difference between passed
                               frame and reference frame, these values are tthresholded
                               to make subsequent defining of center of mass more
                               reliable. [float between 0-100]
                "use_window" : Will window surrounding prior location be
                               imposed?  Allows changes in area surrounding animal"s
                               location on previous frame to be more heavily influential
                               in determining animal"s current location.
                               After finding pixel-wise difference between passed frame
                               and reference frame, difference values outside square window
                               of prior location will be multiplied by (1 - window_weight),
                               reducing their overall influence. [bool]
                "window_size" : If `use_window=True`, the length of one side of square
                                window, in pixels. [uint]
                "window_weight" : 0-1 scale for window, if used, where 1 is maximal
                                  weight of window surrounding prior locaiton.
                                  [float between 0-1]
                "method" : "abs", "light", or "dark".  If "abs", absolute difference
                           between reference and current frame is taken, and thus the
                           background of the frame doesn"t matter. "light" specifies that
                           the animal is lighter than the background. "dark" specifies that
                           the animal is darker than the background.
                "rmv_wire" : True/False, indicating whether to use wire removal function.  [bool]
                "wire_krn" : size of kernel used for morphological opening to remove wire. [int]

         accept_p_frames::[bool]
            Dictates whether to allow videos with temporal compresssion.  Currenntly, if
            more than 1/100 frames returns false, error is flagged.

    -------------------------------------------------------------------------------------
    Returns:
        summary_all:: [pandas.dataframe]
            Pandas dataframe with distance travelled and proportional time spent in each
            region of interest according to user defined time bins, as well as video
            information and parameter values. If no region names are supplied
            (`region_names=None`), only distance travelled will be included.

        layout:: [hv.Layout]
            Holoviews layout wherein for each session the reference frame is returned
            with the regions of interest highlightted and the animals location across
            the session overlaid atop the reference image.

    -------------------------------------------------------------------------------------
    Notes:

    """

    images = []
    for fn in video_dict["FileNames"]:

        print ("Processing File: {f}".format(f=str(fn)))
        video_dict["file"] = fn
        #video_dict["fpath"] = os.path.join(os.path.normpath(video_dict["dpath"]), fn)

        #Print video information. Note that max frame is updated later if fewer frames detected
        cap = cv2.VideoCapture(video_dict["fpath"])
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frames: {frames}".format(frames=cap_max))
        print("nominal fps: {fps}".format(fps=cap.get(cv2.CAP_PROP_FPS)))
        print("dimensions (h x w): {h},{w}".format(
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

        #check for video p-frames
        if accept_p_frames is False:
            check_p_frames(cap)

        video_dict["reference"], image = Reference(video_dict,num_frames=50)
        location = TrackLocation(video_dict,tracking_params)
        location.to_csv(video_dict["output_path"].resolve()/"_LocationOutput.csv", index=False)
        file_summary = Summarize_Location(location, video_dict, bin_dict=bin_dict)

        try:
            summary_all = pd.concat([summary_all,file_summary],sort=False)
        except NameError:
            summary_all = file_summary

        trace = showtrace(video_dict,location)
        heatmap = Heatmap(video_dict, location, sigma=None)
        images = images + [(trace.opts(title=str(fn))), (heatmap.opts(title=str(fn)))]

    #Write summary data to csv file
    sum_pathout = video_dict["dpath"].resolve()/"BatchSummary.csv"
    summary_all.to_csv(sum_pathout, index=False)

    layout = hv.Layout(images)
    return summary_all, layout





########################################################################################

def PlayVideo(video_dict,display_dict,location):
    """
    -------------------------------------------------------------------------------------

    Play portion of video back, displaying animal"s estimated location. Video is played
    in notebook

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        display_dict:: [dict]
            Dictionary with the following keys:
                "start" : start point of video segment in frames [int]
                "end" : end point of video segment in frames [int]
                "resize" : Default is None, in which original size is retained.
                           Alternatively, set to tuple as follows: (width,height).
                           Because this is in pixel units, must be integer values.
                "fps" : frames per second of video file/files to be processed [int]
                "save_video" : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video
                               is something else

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
            Additionally, for each region of interest, boolean array indicating whether
            animal is in the given region for each frame.


    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned

    -------------------------------------------------------------------------------------
    Notes:

    """


    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict["fpath"])#set file\
    if display_dict["save_video"]==True:
        ret, frame = cap.read() #read frame
        frame = process_frame(frame, video_dict)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*"jpeg") #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(str(video_dict["output_path"]/"video_output.avi"),
                                 fourcc, 20.0,
                                 (width, height),
                                 isColor=False)

    #Initialize video play options
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict["start"]+display_dict["start"])

    #Play Video
    for f in range(display_dict["start"],display_dict["stop"]):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = process_frame(frame, video_dict)
            markposition = (int(location["X"][f]),int(location["Y"][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            display_image(frame,display_dict["fps"],display_dict["resize"])
            #Save video (if desired).
            if display_dict["save_video"]==True:
                writer.write(frame)
        if ret == False:
            print("warning. failed to get video frame")

    #Close video window and video writer if open
    print("Done playing segment")
    if display_dict["save_video"]==True:
        writer.release()

def SaveVideo(video_dict,save_dict,location):
    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(str(video_dict["fpath"]))#set file\
    ret, frame = cap.read() #read frame
    frame = process_frame(frame, video_dict)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    #fourcc = 0#cv2.VideoWriter_fourcc(*"jpeg") #only writes up to 20 fps, though video read can be 30.
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    writer = cv2.VideoWriter(str(video_dict["output_path"].resolve()/save_dict["file"]),
                             fourcc, save_dict["fps"],
                             (width, height),
                             isColor=False)

    #Initialize video play options
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict["start"]+save_dict["start"])

    #Play Video
    t_start = datetime.datetime.now()
    for f in range(save_dict["start"],save_dict["stop"]):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = process_frame(frame, video_dict)
            markposition = (int(location["X"][f]),int(location["Y"][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            #display_image(frame,display_dict["fps"],display_dict["resize"])
            writer.write(frame)
        if ret == False:
            print("warning. failed to get video frame")

    #Close video window and video writer if open
    print(datetime.datetime.now()-t_start)
    print("Done saving segment")
    writer.release()

def SaveVideo_fast(video_dict, save_dict, location):
    """
    Faster video saver that overlays a '+' at (X,Y) for each frame using precomputed arrays.

    Inputs:
      - video_dict: expects keys "fpath", "dsmpl", "crop"
      - save_dict: expects keys "file" (output filename), "start", "stop", "fps"
      - location: DataFrame with columns ["Frame", "X", "Y"]; Frame=0 corresponds to absolute frame=video_dict["start"]
    """
    import cv2, numpy as np
    from pathlib import Path

    fpath = str(video_dict["fpath"])
    start_abs = int(video_dict["start"]) + int(save_dict["start"])
    stop_abs  = int(video_dict["start"]) + int(save_dict["stop"])   # exclusive
    dsmpl = float(video_dict.get("dsmpl", 1.0))
    crop  = video_dict.get("crop", None)

    # Precompute location arrays for fast indexing: absolute frame n -> row = n - video_dict["start"]
    # Bound-check so we don't index outside 'location'
    loc_start_abs = int(video_dict["start"])
    loc_end_abs   = loc_start_abs + len(location)  # exclusive

    # Build fast access arrays
    X_arr = location["X"].to_numpy(dtype=np.float32, copy=False)
    Y_arr = location["Y"].to_numpy(dtype=np.float32, copy=False)

    # Open input to get output size after preprocess
    cap0 = cv2.VideoCapture(fpath)
    if not cap0.isOpened():
        raise RuntimeError(f"Could not open input: {fpath}")

    # Seek to the very first frame we will write (to get shape with same preprocess path)
    cap0.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_abs))
    ok, frame0 = cap0.read()
    if not ok or frame0 is None:
        cap0.release()
        raise RuntimeError("Failed to read first frame for shape init")

    # Preprocess first frame to get output size
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    if dsmpl < 1.0:
        gray0 = cv2.resize(
            gray0,
            (int(gray0.shape[1] * dsmpl), int(gray0.shape * dsmpl)),
            cv2.INTER_NEAREST
        )
    gray0 = cropframe(gray0, crop)
    H, W = gray0.shape[:2]

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*"X264")  # change if needed
    out_path = str(Path(video_dict["output_path"]).resolve() / save_dict["file"])
    writer = cv2.VideoWriter(out_path, fourcc, float(save_dict["fps"]), (W, H), isColor=False)
    if not writer.isOpened():
        cap0.release()
        raise RuntimeError(f"Could not open output for writing: {out_path}")

    # Re-use buffers to reduce allocations
    # Marker parameters (small cross)
    cross_half = 15   # half-length of arms: total size ~7px
    color = 255      # grayscale white
    thickness = 1

    # Iterate frames
    n_abs = start_abs
    cap0.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_abs))
    t_start = datetime.datetime.now()
    while n_abs < stop_abs:
        ok, frame_bgr = cap0.read()
        if not ok or frame_bgr is None:
            break

        # Preprocess to match analysis shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if dsmpl < 1.0:
            gray = cv2.resize(
                gray,
                (int(gray.shape[1] * dsmpl), int(gray.shape * dsmpl)),
                cv2.INTER_NEAREST
            )
        gray = cropframe(gray, crop)

        # Map absolute frame to location row
        row = n_abs - loc_start_abs
        if 0 <= row < len(X_arr):
            # Draw cross if coordinates valid
            x = int(round(X_arr[row]))
            y = int(round(Y_arr[row]))
            if 0 <= x < W and 0 <= y < H:
                # Horizontal line
                x0 = max(0, x - cross_half); x1 = min(W - 1, x + cross_half)
                if x1 >= x0:
                    cv2.line(gray, (x0, y), (x1, y), color, thickness, lineType=cv2.LINE_AA)
                # Vertical line
                y0 = max(0, y - cross_half); y1 = min(H - 1, y + cross_half)
                if y1 >= y0:
                    cv2.line(gray, (x, y0), (x, y1), color, thickness, lineType=cv2.LINE_AA)

        writer.write(gray)
        n_abs += 1

    writer.release()
    cap0.release()
    print(datetime.datetime.now()-t_start)
    print("Done saving segment")


def display_image(frame,fps,resize):
    img = PIL.Image.fromarray(frame, "L")
    img = img.resize(size=resize) if resize else img
    buffer = BytesIO()
    img.save(buffer,format="JPEG")
    display(Image(data=buffer.getvalue()))
    time.sleep(1/fps)
    clear_output(wait=True)





########################################################################################

def PlayVideo_ext(video_dict,display_dict,location,crop=None):
    """
    -------------------------------------------------------------------------------------

    Play portion of video back, displaying animal"s estimated location

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        display_dict:: [dict]
            Dictionary with the following keys:
                "start" : start point of video segment in frames [int]
                "end" : end point of video segment in frames [int]
                "fps" : frames per second of video file/files to be processed [int]
                "save_video" : option to save video if desired [bool]
                               Currently, will be saved at 20 fps even if video
                               is something else

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.
            Additionally, for each region of interest, boolean array indicating whether
            animal is in the given region for each frame.


    -------------------------------------------------------------------------------------
    Returns:
        Nothing returned

    -------------------------------------------------------------------------------------
    Notes:

    """

    #Load Video and Set Saving Parameters
    cap = cv2.VideoCapture(video_dict["fpath"])#set file\
    if display_dict["save_video"]==True:
        ret, frame = cap.read() #read frame
        frame = process_frame(frame, video_dict)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        fourcc = 0#cv2.VideoWriter_fourcc(*"jpeg") #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter(str(video_dict["dpath"].resolve()/"video_output.avi"),
                                 fourcc, 20.0,
                                 (width, height),
                                 isColor=False)

    #Initialize video play options
    cap.set(cv2.CAP_PROP_POS_FRAMES,video_dict["start"]+display_dict["start"])
    rate = int(1000/display_dict["fps"])

    #Play Video
    for f in range(display_dict["start"],display_dict["stop"]):
        ret, frame = cap.read() #read frame
        if ret == True:
            frame = process_frame(frame, video_dict)
            markposition = (int(location["X"][f]),int(location["Y"][f]))
            cv2.drawMarker(img=frame,position=markposition,color=255)
            cv2.imshow("preview",frame)
            cv2.waitKey(rate)
            #Save video (if desired).
            if display_dict["save_video"]==True:
                writer.write(frame)
        if ret == False:
            print("warning. failed to get video frame")

    #Close video window and video writer if open
    cv2.destroyAllWindows()
    _=cv2.waitKey(1)
    if display_dict["save_video"]==True:
        writer.release()





########################################################################################

def showtrace(video_dict, location, color="red",alpha=.8,size=3):
    """
    -------------------------------------------------------------------------------------

    Create image where animal location across session is displayed atop reference frame

    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.


        color:: [str]
            Color of trace.  See Holoviews documentation for color options

        alpha:: [float]
            Alpha of trace.  See Holoviews documentation for details

        size:: [float]
            Size of trace.  See Holoviews documentation for details.

    -------------------------------------------------------------------------------------
    Returns:
        holoviews.Overlay
            Location of animal superimposed upon reference. If poly_stream is passed
            than regions of interest will also be outlined.

    -------------------------------------------------------------------------------------
    Notes:

    """

    video_dict["roi_stream"] = video_dict["roi_stream"] if "roi_stream" in video_dict else None
    if video_dict["roi_stream"] != None:
        lst = []
        for poly in range(len(video_dict["roi_stream"].data["xs"])):
            x = np.array(video_dict["roi_stream"].data["xs"][poly]) #x coordinates
            y = np.array(video_dict["roi_stream"].data["ys"][poly]) #y coordinates
            lst.append( [ (x[vert],y[vert]) for vert in range(len(x)) ] )
        poly = hv.Polygons(lst).opts(fill_alpha=0.1,line_dash="dashed")

    #baseframe = cropframe(video_dict["f0"], video_dict["crop"])
    # baseframe = process_frame(video_dict["f0"], video_dict)
    baseframe = video_dict["f0"]
    image = hv.Image((np.arange(baseframe.shape[1]),
                      np.arange(baseframe.shape[0]),
                      baseframe)
                    ).opts(width=int(baseframe.shape[1]*video_dict["stretch"]["width"]),
                           height=int(baseframe.shape[0]*video_dict["stretch"]["height"]),
                           invert_yaxis=True,cmap="gray",toolbar="below",
                           title="Motion Trace")

    points = hv.Scatter(np.array([location["X"],location["Y"]]).T).opts(color="red",alpha=alpha,size=size)

    return (image*poly*points) if video_dict["roi_stream"]!=None else (image*points)





########################################################################################

def Heatmap (video_dict, location, sigma=None):
    """
    -------------------------------------------------------------------------------------

    Create heatmap of relative time in each location. Max value is set to maxiumum
    in any one location.

    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        location:: [pandas.dataframe]
            Pandas dataframe with frame by frame x and y locations,
            distance travelled, as well as video information and parameter values.

        sigma:: [numeric]
            Optional number specifying sigma of guassian filter


    -------------------------------------------------------------------------------------
    Returns:
        map_i:: [holoviews.Image]
            Heatmap image

    -------------------------------------------------------------------------------------
    Notes:
        stretch only affects display

    """
    #baseframe = cropframe(video_dict["f0"], video_dict["crop"])
    # baseframe = process_frame(video_dict["f0"], video_dict)
    baseframe = video_dict["f0"]
    heatmap = np.zeros(baseframe.shape)
    for frame in range(len(location)):
        Y,X = int(location.Y[frame]), int(location.X[frame])
        heatmap[Y,X]+=1

    sigma = np.mean(heatmap.shape)*.05 if sigma == None else sigma
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma)
    heatmap = (heatmap / heatmap.max())*255

    map_i = hv.Image((np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), heatmap))
    map_i.opts(width=int(heatmap.shape[1]*video_dict["stretch"]["width"]),
           height=int(heatmap.shape[0]*video_dict["stretch"]["height"]),
           invert_yaxis=True, cmap="jet", alpha=1,
           colorbar=False, toolbar="below", title="Heatmap")

    return map_i





########################################################################################

def DistanceTool(video_dict):
    """
    -------------------------------------------------------------------------------------

    Creates interactive tool for measuring length between two points, in pixel units, in
    order to ease process of converting pixel distance measurements to some other scale.
    Use point drawing tool to calculate distance beteen any two popints.

    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]


    -------------------------------------------------------------------------------------
    Returns:
        image * points * dmap:: [holoviews.Overlay]
            Reference frame that can be drawn upon to define 2 points, the distance
            between which will be measured and displayed.

        distance:: [dict]
            Dictionary with the following keys:
                "d" : Euclidean distance between two reference points, in pixel units,
                      rounded to thousandth. Returns None if no less than 2 points have
                      been selected.

    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation

    """

    # baseframe = cropframe(video_dict["f0"], video_dict["crop"])
    # baseframe = process_frame(video_dict["f0"], video_dict)
    baseframe = video_dict["f0"]
    #Make reference image the base image on which to draw
    image = hv.Image((
        np.arange(baseframe.shape[1]),
        np.arange(baseframe.shape[0]),
        baseframe))
    image.opts(width=int(baseframe.shape[1]*video_dict["stretch"]["width"]),
               height=int(baseframe.shape[0]*video_dict["stretch"]["height"]),
              invert_yaxis=True,cmap="gray",
              colorbar=True,
               toolbar="below",
              title="Select Points")

    #Create Point instance on which to draw and connect via stream to pointDraw drawing tool
    points = hv.Points([]).opts(active_tools=["point_draw"], color="red",size=10)
    pointDraw_stream = streams.PointDraw(source=points,num_objects=2)

    def markers(data, distance):
        try:
            x_ls, y_ls = data["x"], data["y"]
        except TypeError:
            x_ls, y_ls = [], []

        x_ctr, y_ctr = np.mean(x_ls), np.mean(y_ls)
        if len(x_ls) > 1:
            x_dist = (x_ls[0] - x_ls[1])
            y_dist = (y_ls[0] - y_ls[1])
            distance["px_distance"] = np.around( (x_dist**2 + y_dist**2)**(1/2), 3)
            text = "{dist} px".format(dist=distance["px_distance"])
        return hv.Labels((x_ctr, y_ctr, text if len(x_ls) > 1 else "")).opts(
            text_color="blue",text_font_size="14pt")

    distance = dict(px_distance=None)
    markers_ptl = fct.partial(markers, distance=distance)
    dmap = hv.DynamicMap(markers_ptl, streams=[pointDraw_stream])
    return (image * points * dmap), distance


########################################################################################

def setScale(distance, scale, scale_dict, overwrite = True):

    """
    -------------------------------------------------------------------------------------

    Updates dictionary with scale information, given the true distance between points
    (e.g. 100), and the scale unit (e.g. "cm")

    -------------------------------------------------------------------------------------
    Args:

        distance :: [numeric]
            The real-world distance between the selected points

        scale :: [string]
            The scale used for defining the real world distance.  Can be any string
            (e.g. "cm", "in", "inch", "stone")

        scale_dict :: [dict]
            Dictionary with the following keys:
                "px_distance" : distance between reference points, in pixels [numeric]
                "true_distance" : distance between reference points, in desired scale
                                   (e.g. cm) [numeric]
                "true_scale" : string containing name of scale (e.g. "cm") [str]
                "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]

    -------------------------------------------------------------------------------------
    Returns:
        scale_dict :: [dict]
                Dictionary with the following keys:
                    "px_distance" : distance between reference points, in pixels [numeric]
                    "true_distance" : distance between reference points, in desired scale
                                       (e.g. cm) [numeric]
                    "true_scale" : string containing name of scale (e.g. "cm") [str]
                    "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
    -------------------------------------------------------------------------------------
    Notes:

    """

    if not scale_dict.get("true_distance", None) or overwrite:
        scale_dict["true_distance"] = distance
    if not scale_dict.get("true_scale", None) or overwrite:
        scale_dict["true_scale"] = scale
    return scale_dict



########################################################################################

def ScaleDistance(video_dict, df=None, column=None):
    """
    -------------------------------------------------------------------------------------

    Adds column to dataframe by multiplying existing column by scaling factor to change
    scale. Used in order to convert distance from pixel scale to desired real world
    distance scale.

    -------------------------------------------------------------------------------------
    Args:

        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        df:: [pandas.dataframe]
            Pandas dataframe with column to be scaled.

        column:: [str]
            Name of column in df to be scaled

    -------------------------------------------------------------------------------------
    Returns:
        df:: [pandas.dataframe]
            Pandas dataframe with column of scaled distance values.

    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation

    """

    if "scale" not in video_dict.keys():
        return df

    if video_dict["scale"]["px_distance"]!= None:
        video_dict["scale"]["factor"] = video_dict["scale"]["true_distance"]/video_dict["scale"]["px_distance"]
        new_column = "_".join(["Distance", video_dict["scale"]["true_scale"]])
        df[new_column] = df[column]*video_dict["scale"]["factor"]
        order = [col for col in df if col not in [column,new_column]]
        order = order + [column,new_column]
        df = df[order]
    else:
        print("Distance between reference points undefined. Cannot scale column: {c}.\
        Returning original dataframe".format(c=column))
    return df



########################################################################################

def Mask_select(video_dict, fstfile=False, clear_history=False):
    """
    -------------------------------------------------------------------------------------

    Creates interactive tool for defining regions of interest, based upon array
    `region_names`. If `region_names=None`, reference frame is returned but no regions
    can be drawn.

    -------------------------------------------------------------------------------------
    Args:
        video_dict:: [dict]
            Dictionary with the following keys:
                "dpath" : directory containing files [str]
                "file" : filename with extension, e.g. "myvideo.wmv" [str]
                "start" : frame at which to start. 0-based [int]
                "end" : frame at which to end.  set to None if processing
                        whole video [int]
                "region_names" : list of names of regions.  if no regions, set to None
                "dsmpl" : proptional degree to which video should be downsampled
                        by (0-1).
                "stretch" : Dictionary used to alter display of frames, with the following keys:
                        "width" : proportion by which to stretch frame width [float]
                        "height" : proportion by which to stretch frame height [float]
                        *Does not influence actual processing, unlike dsmpl.
                "reference": Reference image that the current frame is compared to. [numpy.array]
                "roi_stream" : Holoviews stream object enabling dynamic selection in response to
                               selection tool. `poly_stream.data` contains x and y coordinates of roi
                               vertices. [hv.streams.stream]
                "crop" : Enables dynamic box selection of cropping parameters.
                         Holoviews stream object enabling dynamic selection in response to
                         `stream.data` contains x and y coordinates of crop boundary vertices.
                         [hv.streams.BoxEdit]
                "mask" : [dict]
                    Dictionary with the following keys:
                        "mask" : boolean numpy array identifying regions to exlude
                                 from analysis.  If no such regions, equal to
                                 None. [bool numpy array)
                        "mask_stream" : Holoviews stream object enabling dynamic selection
                                in response to selection tool. `mask_stream.data` contains
                                x and y coordinates of region vertices. [holoviews polystream]
                "scale:: [dict]
                        Dictionary with the following keys:
                            "px_distance" : distance between reference points, in pixels [numeric]
                            "true_distance" : distance between reference points, in desired scale
                                               (e.g. cm) [numeric]
                            "true_scale" : string containing name of scale (e.g. "cm") [str]
                            "factor" : ratio of desired scale to pixel (e.g. cm/pixel [numeric]
                "ftype" : (only if batch processing)
                          video file type extension (e.g. "wmv") [str]
                "FileNames" : (only if batch processing)
                              List of filenames of videos in folder to be batch
                              processed.  [list]
                "f0" : (only if batch processing)
                        first frame of video [numpy array]

        fstfile:: [bool]
            Dictates whether to use first file in video_dict["FileNames"] to generate
            reference.  True/False

    -------------------------------------------------------------------------------------
    Returns:
        image * poly * dmap:: [holoviews.Overlay]
            First frame of video that can be drawn upon to define regions of interest.

        mask:: [dict]
            Dictionary with the following keys:
                "mask" : boolean numpy array identifying regions to exlude
                         from analysis.  If no such regions, equal to
                         None. [bool numpy array)
                "mask_stream" : Holoviews stream object enabling dynamic selection
                        in response to selection tool. `mask_stream.data` contains
                        x and y coordinates of region vertices. [holoviews polystream]

    -------------------------------------------------------------------------------------
    Notes:
        - if `stretch` values are modified, this will only influence dispplay and not
          calculation

    """

    #Load first file if batch processing
    if fstfile:
        video_dict["file"] = video_dict["FileNames"][0]
        #video_dict["fpath"] = os.path.join(os.path.normpath(video_dict["dpath"]), video_dict["file"])
        if video_dict["fpath"].exists():
            print("file: {file}".format(file=str(video_dict["fpath"])))
            cap = cv2.VideoCapture(str(video_dict["fpath"]))
        else:
            raise FileNotFoundError("{file} not found. Check that directory and file names are correct".format(
                file=str(video_dict["fpath"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict["start"])
        ret, frame = cap.read()
        frame = process_frame(frame, video_dict,
                              do_gray = True, do_angle = False, do_dsmpl = True, do_crop = False)
        video_dict["f0"] = frame

    #Make first image the base image on which to draw
    # f0 = cropframe(
    #     video_dict["f0"],
    #     video_dict.get("crop")
    # )
    # f0 = process_frame(video_dict["f0"], video_dict)
    f0 = video_dict["f0"]
    image = hv.Image((np.arange(f0.shape[1]), np.arange(f0.shape[0]), f0))
    image.opts(width=int(f0.shape[1]*video_dict["stretch"]["width"]),
               height=int(f0.shape[0]*video_dict["stretch"]["height"]),
              invert_yaxis=True,cmap="gray",
              colorbar=True,
               toolbar="below",
              title="Draw Regions to be Exluded")

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    if "mask" not in video_dict:
        video_dict["mask"] = {"mask": None,
                              "stream": None}
                              #"point_stream": None}
    mask = video_dict["mask"]
    poly_dicts = []
    if (
        (isinstance(mask["stream"], streams.PolyDraw) or isinstance(mask["stream"], DataStub)) and
        (mask["stream"].data.get("xs", None) or mask["stream"].data.get("x", None)) and
        (not clear_history)
       ):
        initial_data = mask["stream"].data
        for i in range(len(initial_data["xs"])):
            poly_dicts.append({
                "x": initial_data["xs"][i],
                "y": initial_data["ys"][i]})

    else:
        initial_data = None
        # mask = dict(mask=None)

    poly = hv.Polygons(poly_dicts)
    mask["stream"] = streams.PolyDraw(source=poly,
                                      drag=True,
                                      show_vertices=True,
                                      data=initial_data)
    #poly_stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=["poly_draw"])
    points = hv.Points([]).opts(active_tools=["point_draw"], color="red",size=10)
    #mask["point_stream"] = streams.PointDraw(source=points,num_objects=2, drag=True)

    def make_mask(data, mask):
        try:
            x_ls, y_ls = data["xs"], data["ys"]
        except TypeError:
            x_ls, y_ls = [], []

        if len(x_ls)>0:
            mask["mask"] = np.zeros(f0.shape)
            for submask in range(len(x_ls)):
                x = np.array(mask["stream"].data["xs"][submask]) #x coordinates
                y = np.array(mask["stream"].data["ys"][submask]) #y coordinates
                xy = np.column_stack((x,y)).astype("uint64") #xy coordinate pairs
                cv2.fillPoly(mask["mask"], pts =[xy], color=1) #fill polygon
            mask["mask"] = mask["mask"].astype("bool")
        return hv.Labels((0,0,""))


    make_mask_ptl = fct.partial(make_mask, mask=mask)
    dmap = hv.DynamicMap(make_mask_ptl, streams=[mask["stream"]])
    return image*poly*dmap, mask



def check_p_frames(cap, p_prop_allowed=.01, frames_checked=300):
    """
    -------------------------------------------------------------------------------------

    Checks whether video contains substantial portion of p/blank frames

    -------------------------------------------------------------------------------------
    Args:
        cap:: [cv2.videocapture]
            OpenCV video capture object.
        p_prop_allowed:: [numeric]
            Proportion of putative p-frames permitted.  Alternatively, proportion of
            frames permitted to return False when grabbed.
        frames_checked:: [numeric]
            Number of frames to scan for p/blank frames.  If video is shorter
            than number of frames specified, will use number of frames in video.

    -------------------------------------------------------------------------------------
    Returns:

    -------------------------------------------------------------------------------------
    Notes:

    """

    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    p_allowed = int(frames_checked*p_prop_allowed)

    p_frms = 0
    for i in range(frames_checked):
        ret, frame = cap.read()
        p_frms = p_frms+1 if ret==False else p_frms
    if p_frms>p_allowed:
        raise RuntimeError(
            "Video compression method not supported. " + \
            "Approximately {p}% frames are p frames or blank. ".format(
                p=(p_frms/frames_checked)*100) + \
            "Consider video conversion.")


########################################################################################
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer("bokeh").get_plot(image).state
#bokeh_obj.output_backend = "svg"
#export_svgs(bokeh_obj, dpath + "/" + "Calibration_Frame.svg")


def _resize_array(img: np.ndarray, scale: float, is_mask=False, dtype=None):
    """
    Resize 2D array by scale using cv2.resize with appropriate interpolation.
    - is_mask=True uses nearest-neighbor.
    - dtype forces output dtype; otherwise preserve.
    """
    if img is None:
        return None
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    out = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out

def _scale_coords_in_polydraw_like(stream_obj, scale: float):
    """
    Scale in-place the coordinates in a PolyDraw-like structure:
    stream_obj.data["xs"] and ["ys"] are lists of np.ndarray.
    """
    if stream_obj is None:
        return
    data = getattr(stream_obj, "data", None)
    if not isinstance(data, dict):
        return
    xs = data.get("xs", [])
    ys = data.get("ys", [])
    if xs is None or ys is None:
        return
    # Scale each ndarray; keep dtype float
    for i in range(len(xs)):
        if xs[i] is not None:
            xs[i] = (np.asarray(xs[i], dtype=np.float64) * scale)
        if ys[i] is not None:
            ys[i] = (np.asarray(ys[i], dtype=np.float64) * scale)
    # Write back
    data["xs"] = xs
    data["ys"] = ys
    stream_obj.data = data  # for DataStub-like behavior

def _scale_boxedit_like(box_obj, scale: float):
    """
    Scale in-place a BoxEdit-like structure with data keys x0,x1,y0,y1 (arrays len=1).
    """
    if box_obj is None:
        return
    data = getattr(box_obj, "data", None)
    if not isinstance(data, dict):
        return
    for k in ("x0","x1","y0","y1"):
        if k in data and isinstance(data[k], (list, tuple, np.ndarray)) and len(data[k]) >= 1:
            v = data[k][0]
            if v is not None:
                new_v = float(v) * scale
            else:
                new_v = None
            # maintain list shape
            if isinstance(data[k], np.ndarray):
                arr = data[k].copy()
                arr[:1] = [new_v]
                data[k] = arr
            else:
                data[k] = [new_v] + list(data[k][1:])
    box_obj.data = data

def change_dsmpl_all(video_dict: dict, new_dsmpl: float) -> dict:
    """
    Rescales all arrays and coordinate streams in video_dict from current dsmpl to new_dsmpl.
    - Scales images (f0, reference), mask["mask"] by resizing.
    - Scales coordinates in crop (BoxEdit), mask["stream"] (PolyDraw), and roi_stream (PolyDraw).
    - Leaves dpath, fpath, etc., untouched.
    Returns the updated video_dict (also modified in place).
    """
    vd = video_dict  # modify in place
    old = float(vd.get("dsmpl", 1.0))
    new = float(new_dsmpl)
    if old <= 0 or new <= 0:
        raise ValueError("dsmpl values must be > 0")
    if np.isclose(old, new):
        vd["dsmpl"] = new
        return vd

    scale = new / old

    # 1) Resize arrays
    # f0: uint8 grayscale (as per your pipeline)
    if "f0" in vd and isinstance(vd["f0"], np.ndarray):
        vd["f0"] = _resize_array(vd["f0"], scale, is_mask=False, dtype=np.uint8)

    # reference: can be int16 per your code, keep dtype
    if "reference" in vd and isinstance(vd["reference"], np.ndarray):
        ref_dtype = vd["reference"].dtype
        # Use linear for smoothness; cast back
        resized = _resize_array(vd["reference"].astype(np.float32), scale, is_mask=False)
        vd["reference"] = np.clip(resized, np.iinfo(ref_dtype).min if ref_dtype.kind in "iu" else -32768,
                                  np.iinfo(ref_dtype).max if ref_dtype.kind in "iu" else 32767).astype(ref_dtype)

    # mask["mask"]: boolean mask; nearest neighbor
    if "mask" in vd and isinstance(vd["mask"], dict):
        m = vd["mask"].get("mask", None)
        if isinstance(m, np.ndarray):
            # Resize via nearest neighbor using uint8 -> back to bool
            m_u8 = m.astype(np.uint8, copy=False)
            m_resized = _resize_array(m_u8, scale, is_mask=True, dtype=np.uint8)
            vd["mask"]["mask"] = (m_resized > 0)

    # 2) Scale coordinate-bearing objects
    # crop: BoxEdit-like with x0,x1,y0,y1
    _scale_boxedit_like(vd.get("crop", None), scale)

    # mask["stream"]: PolyDraw-like
    if "mask" in vd and isinstance(vd["mask"], dict):
        _scale_coords_in_polydraw_like(vd["mask"].get("stream", None), scale)

    # roi_stream: PolyDraw-like
    _scale_coords_in_polydraw_like(vd.get("roi_stream", None), scale)

    # 3) Update dsmpl
    vd["dsmpl"] = new

    return vd
