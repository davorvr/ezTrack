### ALL USER CONFIGURATION IS AT THE BOTTOM, IN THE BLOCK
### if __name__ == "__main__": 

import holoviews as hv
import numpy as np
import pandas as pd
import LocationTracking_Functions as lt
from pathlib import Path
import cv2
import pickle
import panel as pn
from multiprocessing import Pool, cpu_count

def create_layout(hv_obj):
    hv_pane = pn.pane.HoloViews(hv_obj)
    layout = pn.Column()
    layout.append(hv_pane)
    return layout

def process_video_file(video_dict):
    global video_dict_override
    global scale
    global heatmap_h
    global heatmap_w
    global tracking_params
    global display_dict
    
    try:
        if video_dict_override:
            video_dict.update(video_dict_override)
    except NameError:
        pass
    video_dict["fpath"] = video_dict["dpath"]/video_dict["file"]
    cap = cv2.VideoCapture(str(video_dict["fpath"]))

    video_dict["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
    video_dict["start"] = int(video_dict["start_s"]*video_dict["fps"])
    video_dict["bin_duration"] = int(video_dict["bin_duration_s"]*video_dict["fps"])
    video_dict["vid_duration"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dict["end"] = int(video_dict["end_s"]*video_dict["fps"]) if video_dict["end_s"] is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    if video_dict["start"] < 0:
        video_dict["start"] += video_dict["vid_duration"]
        video_dict["start_s"] += video_dict["vid_duration"]/video_dict["fps"]

    video_dict["fname_stem"] = Path(video_dict["file"]).stem
    video_dict["output_path"] = video_dict["dpath"]/video_dict["fname_stem"]
    video_dict["output_path"].mkdir(exist_ok=True)

    if video_dict["bins_s"]:
        video_dict["bins"] = {}
        for i, b in enumerate(video_dict["bins_s"]):
            video_dict["bins"][str(i+1)] = (b[0]*video_dict["fps"], b[1]*video_dict["fps"])
    elif video_dict["bin_duration"] and not video_dict["bins_s"]:
        video_dict["bins"] = {}
        for i, bin_start in enumerate(range(video_dict["start"], video_dict["vid_duration"], video_dict["bin_duration"])):
            video_dict["bins"][str(i+1)] = (bin_start, bin_start+video_dict["bin_duration"])
    video_dict["last_frame"] = video_dict["vid_duration"]-video_dict["start"]
    video_dict["full_bin"] = [(video_dict["start"], video_dict["last_frame"])]
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict["start"])
    ret, frame = cap.read()
    frame = lt.process_frame(frame, video_dict,
                             do_gray = True, do_angle = True, do_dsmpl = True, do_crop = True)
    video_dict["f0"] = frame

    #%%output size = 300
    ### NE ZABORAVI ME UPALITI KASNIJE
    #video_dict['reference'], img_ref = lt.Reference(video_dict, num_frames=50)
    #video_dict['reference'], img_ref = lt.Reference(video_dict, frames=np.arange(100,2000,10))
    #video_dict['reference'], img_ref = lt.Reference(video_dict, segment=(200,1000))

    # hv.save(img_ref, video_dict["output_path"]/str(video_dict["fname_stem"]+"_reference.png"), fmt='png')

    #layout = create_layout(img_ref)
    #layout

    viewpane_dimensions = {"x": abs(video_dict["crop"].data["x1"][0]-video_dict["crop"].data["x0"][0]),
                           "y": abs(video_dict["crop"].data["y1"][0]-video_dict["crop"].data["y0"][0])}
    max_x = viewpane_dimensions["x"]
    max_y = viewpane_dimensions["y"]
    if video_dict["region_names"] == "OF":
        roi_size = video_dict["OF_preset_config"]["wall_fraction"]
        roi_width = roi_size*viewpane_dimensions["x"]
        roi_height = roi_size*viewpane_dimensions["y"]
        if video_dict["OF_preset_config"]["calculate_against"] == "height":
            roi_width = roi_height
        elif video_dict["OF_preset_config"]["calculate_against"] == "width":
            roi_height = roi_width
        #['wall_L','wall_R','wall_T','wall_B', 'corner_UL', 'corner_UR', 'corner_BL', 'corner_BR', 'center'],
        rois = { "wall_L" : {"xs": [0, 0, roi_width, roi_width],
                            "ys": [0, max_y, max_y, 0]},
                "wall_R" : {"xs": [max_x-roi_width, max_x-roi_width, max_x, max_x],
                            "ys": [0, max_y, max_y, 0]},
                "wall_T" : {"xs": [0, 0, max_x, max_x],
                            "ys": [0, roi_height, roi_height, 0]},
                "wall_B" : {"xs": [0, 0, max_x, max_x],
                            "ys": [max_y, max_y-roi_height, max_y-roi_height, max_y]},
                "corner_UL" : {"xs": [0, 0, roi_width, roi_width],
                                "ys": [0, roi_height, roi_height, 0]},
                "corner_UR" : {"xs": [max_x, max_x-roi_width, max_x-roi_width, max_x],
                                "ys": [0, 0, roi_height, roi_height]},
                "corner_BL" : {"xs": [0, roi_width, roi_width, 0],
                                "ys": [max_y, max_y, max_y-roi_height, max_y-roi_height]},
                "corner_BR" : {"xs": [max_x, max_x-roi_width, max_x-roi_width, max_x],
                                "ys": [max_y, max_y, max_y-roi_height, max_y-roi_height]},
                "center" : {"xs": [roi_width, max_x-roi_width, max_x-roi_width, roi_width],
                            "ys": [roi_height, roi_height, max_y-roi_height, max_y-roi_height]} }
        rois_poly = []
        for roi, coords in rois.items():
            rois_poly.append({("x", "y"):list(zip(coords["xs"], coords["ys"]))})
            #rois_poly.append({("xs", "ys"):[coords["xs"], coords["ys"]]})
        rois_poly = hv.Polygons(data=rois_poly)
        #print(rois_poly)
        roi_data = {"xs": [d["xs"] for d in rois.values()],
                    "ys": [d["ys"] for d in rois.values()]}

        video_dict["roi_stream"] = lt.DataStub(roi_data)
    elif video_dict["region_names"] == "EPM":
        centre_x = viewpane_dimensions["x"]/2+video_dict["EPM_preset_config"]["centre_offset_x"]
        centre_y = viewpane_dimensions["y"]/2+video_dict["EPM_preset_config"]["centre_offset_y"]
        roi_halfwidth = video_dict["EPM_preset_config"]["vertical_arm_width_frac"]*viewpane_dimensions["x"]/2
        roi_halfheight = video_dict["EPM_preset_config"]["horizontal_arm_height_frac"]*viewpane_dimensions["y"]/2
        #['open_T', 'open_B', 'closed_L', 'closed_R', 'center']
        rois = { "open_T" : {"xs": [centre_x-roi_halfwidth, centre_x-roi_halfwidth, centre_x+roi_halfwidth, centre_x+roi_halfwidth],
                            "ys": [0, centre_y-roi_halfheight, centre_y-roi_halfheight, 0]},
                "open_B" : {"xs": [centre_x-roi_halfwidth, centre_x-roi_halfwidth, centre_x+roi_halfwidth, centre_x+roi_halfwidth],
                            "ys": [max_y, centre_y+roi_halfheight, centre_y+roi_halfheight, max_y]},
                "closed_L" : {"xs": [0, centre_x-roi_halfwidth, centre_x-roi_halfwidth, 0],
                            "ys": [centre_y-roi_halfheight, centre_y-roi_halfheight, centre_y+roi_halfheight, centre_y+roi_halfheight]},
                "closed_R" : {"xs": [max_x, centre_x+roi_halfwidth, centre_x+roi_halfwidth, max_x],
                            "ys": [centre_y-roi_halfheight, centre_y-roi_halfheight, centre_y+roi_halfheight, centre_y+roi_halfheight]},
                "center" : {"xs": [centre_x-roi_halfwidth, centre_x-roi_halfwidth, centre_x+roi_halfwidth, centre_x+roi_halfwidth],
                            "ys": [centre_y-roi_halfheight, centre_y+roi_halfheight, centre_y+roi_halfheight, centre_y-roi_halfheight]}
        }
        mask = { "UL" : {"xs": [0, centre_x-roi_halfwidth, centre_x-roi_halfwidth, 0],
                        "ys": [0, 0, centre_y-roi_halfheight, centre_y-roi_halfheight]},
                "UR" : {"xs": [max_x, centre_x+roi_halfwidth, centre_x+roi_halfwidth, max_x],
                        "ys": [0, 0, centre_y-roi_halfheight, centre_y-roi_halfheight]},
                "BL" : {"xs": [0, centre_x-roi_halfwidth, centre_x-roi_halfwidth, 0],
                        "ys": [centre_y+roi_halfheight, centre_y+roi_halfheight, max_y, max_y]},
                "BR" : {"xs": [max_x, centre_x+roi_halfwidth, centre_x+roi_halfwidth, max_x],
                        "ys": [centre_y+roi_halfheight, centre_y+roi_halfheight, max_y, max_y]}
        }
        #rois_poly = []
        #for roi, coords in rois.items():
        #    rois_poly.append({("x", "y"):list(zip(coords["xs"], coords["ys"]))})
        #    #rois_poly.append({("xs", "ys"):[coords["xs"], coords["ys"]]})
        #rois_poly = hv.Polygons(data=rois_poly)
        #print(rois_poly)

        roi_data = {"xs": [d["xs"] for d in rois.values()],
                    "ys": [d["ys"] for d in rois.values()]}
        video_dict["roi_stream"] = lt.DataStub(roi_data)

        mask_data = {"xs": [d["xs"] for d in mask.values()],
                    "ys": [d["ys"] for d in mask.values()]}
        mask_bool = np.zeros(video_dict["f0"].shape)
        for submask in range(len(mask_data["xs"])):
            x = np.array(mask_data["xs"][submask]) #x coordinates
            y = np.array(mask_data["ys"][submask]) #y coordinates
            xy = np.column_stack((x,y)).astype("uint64") #xy coordinate pairs
            cv2.fillPoly(mask_bool, pts =[xy], color=1) #fill polygon
        mask_bool = mask_bool.astype("bool")
        video_dict["mask"] = {
            "stream": lt.DataStub(mask_data),
            "mask": mask_bool
        }

    #%%output size = 300
    # h, w = lt.cropframe(video_dict["f0"], video_dict["crop"]).shape
    # scale = 3.0
    try:
        if video_dict["roi_stream"]:
            img_roi, video_dict['roi_stream'] = lt.ROI_plot(video_dict)
            # img_roi.opts(width=int(w*scale)+80, height=int(h*scale))
            hv.save(img_roi, video_dict["output_path"]/str(video_dict["fname_stem"]+"_ROIs.png"), fmt='png')
            #layout = create_layout(img_roi)
            #layout
    except KeyError:
        pass

    try:
        if video_dict["mask"]:
            img_mask, video_dict['mask'] = lt.Mask_select(video_dict)
            hv.save(img_mask, video_dict["output_path"]/str(video_dict["fname_stem"]+"_donottrack.png"), fmt='png')
    except KeyError:
        pass

    #distance = 47.376
    #scale = 'cm'
    #video_dict["scale"] = {"px_distance": np.sqrt(video_dict["reference"].shape[0]**2+video_dict["reference"].shape[1]**2)}
    try:
        if scale_override:
            video_dict["scale"].update(scale_override)
        elif scale_override is None:
            pass
        elif scale_override is False:
            try:
                del video_dict["scale"]
            except KeyError:
                pass
    except NameError:
        pass
    
    #video_dict['scale'] = lt.setScale(distance, scale, video_dict['scale'], overwrite = False)

    # %%output size = 70

    img_exmpls = lt.LocationThresh_View(video_dict, tracking_params, examples=16)
    # scale = 3.0
    # img_exmpls.opts(width=int(w*scale), height=int(h*scale))
    hv.save(img_exmpls, video_dict["output_path"]/str(video_dict["fname_stem"]+"_CheckTracking.png"), fmt='png')

    with open(video_dict["output_path"]/"video_dict.pickle", 'wb') as f:
        pickle.dump(lt.copy_video_dict(video_dict), f)

    location = lt.TrackLocation(video_dict, tracking_params)
    #if __name__ == "__main__":
    #    location = lt.TrackLocation_parallel(video_dict, tracking_params)
    location.to_csv(video_dict["output_path"]/str(video_dict["fname_stem"]+'_LocationOutput.csv'), index=False)

    # %%output size = 100

    # scale = 3.0
    plt_dist = hv.Curve((location['Frame'],location['Distance_px']),'Frame','Pixel Distance').opts(
        height=heatmap_h,width=heatmap_w,color='red',title="Distance Across Session",toolbar="below")
    plt_trks = lt.showtrace(video_dict, location, color="red", alpha=.05, size=2)
    # plt_trks.opts(width=int(w*scale), height=int(h*scale))
    plt_hmap = lt.Heatmap(video_dict, location, sigma=sigma)
    # plt_hmap.opts(width=int(w*scale), height=int(h*scale))
    trace_img = (plt_trks + plt_hmap + plt_dist).cols(1)
    hv.save(trace_img, video_dict["output_path"]/str(video_dict["fname_stem"]+"_TraceAndHeatmap.png"), fmt='png')

    #bin_dict = {
    #    '1' : (0,10),
    #    '2' : (10,20),
    #    '3' : (20,30)
    #}

    summary_binned = lt.Summarize_Location(location, video_dict, bin_dict=video_dict["bins"])
    summary_binned.to_csv(video_dict["output_path"]/str(video_dict["fname_stem"]+'_SummaryStats_binned.csv'), index=False)
    summary = lt.Summarize_Location(location, video_dict, bin_dict=video_dict["full_bin"])
    # summary_full_filename = video_dict["dpath"]/str(video_dict["dpath"].stem+'_SummaryStats.csv')
    # if not summary_full_filename.exists():
    #     summary_full = summary
    # else:
    #     summary_full = pd.read_csv(summary_full_filename)
    #     if summary_full.loc[summary_full["File"] == video_dict["file"]].empty:
    #         summary_full = pd.concat([summary_full, summary])
    #     else:
    #         summary_full = summary_full.set_index("File")
    #         summary_full.update(summary.set_index("File"))
    #         summary_full = summary_full.reset_index()
    # summary_full.to_csv(summary_full_filename, index=False)

    # Display parameters (used to generate the video):
    if save_video:
        display_dict = {
            'start'      : 0,
            'stop'       : video_dict["last_frame"]-video_dict["start"],
            'fps'        : video_dict["fps"],
            'resize'     : None,
            'file'       : video_dict["fname_stem"]+"_tracked.mkv"
        }
        lt.SaveVideo(video_dict,display_dict,location)

    #with open(video_dict_storefile, 'wb') as f:
    #    pickle.dump(lt.copy_video_dict(video_dict), f)

    return video_dict["file"], summary

if __name__ == "__main__":

    #### CONFIGURATION ####

    # Path where the videos and the configs are located:
    dpath = "/home/davor/ext/3/OF"
    dpath = Path(dpath)

    # Define how many configs we're going to use, and their names:
    video_dicts = ["video_dict_ortho",
                   "video_dict_rotated"]
    # Specify the exact config file to load for each of the configs:
    video_dicts_storefiles = {"video_dict_ortho": dpath/"video_dict_storefile_GXortho.pickle",
                              "video_dict_rotated": dpath/"video_dict_storefile_GXrotated.pickle"}
    # Specify a list of videos for each configuration file. The example code uses two types of
    # automated matching or list generation, but the specification can be as simple as:
    # video_dicts_filelists = {"video_dict_ortho": ["1.mp4", "2.mp4"],
    #                          "video_dict_rotated": ["3.mp4", "4.mp4"]}
    video_dicts_filelists = {"video_dict_ortho": list(dpath.glob("GX0104[4 5 6 7]?.MP4"))+[dpath/"GX010480.MP4"],
                             "video_dict_rotated": [dpath/f"GX01048{i}.MP4" for i in range(1,6)]}

    ### Another example:
    # video_dicts = ["bottom_left",
    #             "bottom_right",
    #             "top_left",
    #             "top_right"]
    # video_dicts_storefiles = {"bottom_left": dpath/"video_dict_bottom_left.pickle",
    #                           "bottom_right": dpath/"video_dict_bottom_right.pickle",
    #                           "top_left": dpath/"video_dict_top_left.pickle",
    #                           "top_right": dpath/"video_dict_top_right.pickle"}
    # video_dicts_filelists = {"bottom_left": list(dpath.glob("*_bottom_left_upd.MP4")),
    #                          "bottom_right": list(dpath.glob("*_bottom_right_upd.MP4")),
    #                          "top_left": list(dpath.glob("*_top_left_upd.MP4")),
    #                          "top_right": list(dpath.glob("*_top_right_upd.MP4"))}

    
    # Note: "override" variables can be commented out to load all values from the
    # configuration file.
    #
    # The default parameters that overwrite those in the loaded video_dict of a
    # specific configuration file. Add parameters to override (for them to apply
    # to all processed files), remove parameters to use those preloaded from the
    # config file. Do NOT override `dpath` and `file` since these are used to 
    # iterate across video files, i.e. batch process - if you want to process
    # single files, use the interactive notebook.
    video_dict_override = {
            #'dpath'         : dpath,
            #'file'          : video_filename.name,
            'start_s'       : -600,
            'end_s'         : None,
            'region_names'  : ['wall_L','wall_R','wall_T','wall_B', 'corner_UL', 'corner_UR', 'corner_BL', 'corner_BR', 'center'],
            'dsmpl'         : 0.5,
            'stretch'       : dict(width=3, height=3),
            'bin_duration_s': 60,
            'angle'         : None
        }
    # Dictionary specifying the pixel-to-cm scale calculation. Specify a `px_distance`
    # in pixels, and its corresponding length in centimetres. If not specified, as:
    #     scale_override = None
    # ...the values will be pulled from video_dict.
    # If:
    #     scale_override = False
    # ...no scale will be used and lengths will be expressed in pixels only.
    scale_override = {
            "px_distance": 400,
            "true_distance": 47.376,
            "true_scale": "cm"
        }
    # Heatmap and trace graphs width and height in pixels
    heatmap_w, heatmap_h = 600, 200
    # Heatmap sigma (blurriness)
    sigma_heatmap = None
    # Tracking parameters (used to adjust the tracking algorithm):
    tracking_params = {
        'loc_thresh'    : 98.0,
        'use_window'    : True,
        'window_size'   : 150,
        'window_weight' : .9,
        'method'        : 'abs',
        'rmv_wire'      : True,
        'wire_krn'      : 5,
        "progress_bar"  : False
    }
    # Whether to save the output video with tracking overlay
    save_video = True

    #### END CONFIGURATION ####
    
    args = []
    for vd_name in video_dicts:
        with open(video_dicts_storefiles[vd_name], 'rb') as f:
            vd = pickle.load(f)
        vd["dpath"] = dpath
        if "angle" not in vd:
            vd["angle"] = None
        for f in video_dicts_filelists[vd_name]:
            vd_copy = lt.copy_video_dict(vd)
            vd_copy["file"] = f.name
            args.append(lt.copy_video_dict(vd_copy))

    with Pool(6) as pool:
        summaries = pool.map(process_video_file, args)
    summary_full_filename = dpath/str(dpath.stem+'_SummaryStats.csv')
    if not summary_full_filename.exists():
        summary_full = pd.concat([s[1] for s in summaries])
    else:
        summary_full = pd.read_csv(summary_full_filename)
        for fname, s in summaries:
            if summary_full.loc[summary_full["File"] == fname].empty:
                summary_full = pd.concat([summary_full, s])
            else:
                summary_full = summary_full.set_index("File")
                summary_full.update(s.set_index("File"))
                summary_full = summary_full.reset_index()
    summary_full.to_csv(summary_full_filename, index=False)
    # print("Finished:", completed)
