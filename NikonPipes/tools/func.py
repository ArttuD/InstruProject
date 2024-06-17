import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 

import tqdm

from nd2reader import ND2Reader
import h5py
import pickle
import json

import datetime

import warnings
warnings.filterwarnings('ignore')

import ffmpeg

from skimage.filters import rank, threshold_otsu, threshold_local#
from skimage import morphology

import scipy 
import skimage

import pandas as pd
import seaborn as sns
import csv

import six
import struct
import re



####
##Functions
####


def parse_raw_dict(day, video_path, own_meta):

    fh = video_path

    parts = os.path.split(video_path)[-1].split("_")
    day = str(parts[0])

    label_map = None

    if isinstance(fh, str):
        if not fh.endswith(".nd2"):
            raise InvalidFileType(
                ("The file %s you want to read with nd2reader" % fh)
                + " does not have extension .nd2."
            )
        
        filename = fh

        fh = open(fh, "rb")

    _fh = fh
    _fh.seek(-8, 2)
    chunk_map_start_location = struct.unpack("Q", _fh.read(8))[0]
    _fh.seek(chunk_map_start_location)
    raw_text = _fh.read(-1)
    label_map = LabelMap(raw_text)
    datasTT = RawMetadata(_fh, label_map)


    seeding_density = []
    well_name = []
    well_info = datasTT.image_metadata[b'SLxExperiment'][b'ppNextLevelEx'][b''][b'uLoopPars'][b'Points'][b'']
    for i in range(len(well_info)):
        
        label = (well_info[i][b'dPosName']).decode("utf8")
        lable_parts = label.split("_")
        if len(lable_parts) == 1:
            seeding_density.append(500)
        else:
            try:
                seeding_density.append(int(lable_parts[1]))
            except:
                seeding_density.append(500)
                print(lable_parts[1])

        well_name.append(lable_parts[0])

    own_meta[day]["cell"] = well_name
    own_meta[day]["seeding_density"] = seeding_density
    own_meta[day]["dt"] = datasTT.image_metadata[b'SLxExperiment'][b'uLoopPars'][b'dPeriod']*1e-3

    return own_meta
        
        
def find_plot_size(num_data):
    plot_ind = 1
    while True:
        if (plot_ind*plot_ind)>=num_data:
            break
        else:
            plot_ind += 1
    return plot_ind

#Incease birghtness if the video is dark (visualization)
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



def Kittler_16(im, out):
    
    max_val = 2**16-1
    h,g = np.histogram(im.ravel(),max_val,[0,max_val])
    h = h.astype("float")
    g = g.astype("float")
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s/c - (m/c)**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
    p =  c / c[-1]
    v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    out[:,:] = 0
    out[im >= t] = max_val
    return out

def yen_filter_16(image):

    max_val = 2**16
    counts, bin_centers =skimage.exposure.histogram(image.reshape(-1), max_val, source_range='image', normalize=False)

    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns ``bin_centers`` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = counts.astype('float32', copy=False) / counts.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf**2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


#Metadata parser
def load_metadata(images):
    meta_dict = {}
     # number of locations start 1
    meta_dict["n_fields"] = images.metadata['fields_of_view'].stop

    #number of timeseteps
    meta_dict["n_frames"] = images.metadata['num_frames']

    
    #meta_dict["z_level"] = (np.max(images.metadata['z_coordinates'])-np.min(images.metadata['z_coordinates']))

    meta_dict["z_level"] =  float(images.metadata["z_coordinates"][:images.metadata["z_levels"].stop][-1]-images.metadata["z_coordinates"][:images.metadata["z_levels"].stop][0])/float(images.metadata["z_levels"].stop)
    #number of levels starting from 1
    meta_dict["n_levels"] = images.metadata['z_levels'].stop
    meta_dict["z_step"] = meta_dict["z_level"] /meta_dict["n_levels"]

    #list of channels
    meta_dict["channels"] = images.metadata['channels']

    #number of channels
    meta_dict["n_channels"] = len(meta_dict["channels"])

    meta_dict["m"] = images.metadata['pixel_microns']
    meta_dict["height"] = images.metadata["height"]
    meta_dict["width"] = images.metadata["width"]

    return meta_dict


#Segmentation
def process_projection(img, x_start, y_start):

    frame = img.copy()
    th_num = 500

    frame = scipy.ndimage.gaussian_filter(frame, (5,5))
    frame = frame.astype(int)

    glob_thresh = threshold_otsu(frame)
    binary_local = frame > glob_thresh

    radius = 30   
    footprint = morphology.disk(radius)
    local_otsu = rank.otsu(frame.copy(), footprint)
    lo = frame >= local_otsu    
    l1 = np.zeros_like(lo)
    l1[binary_local] = lo[binary_local]

    # could/should be improved    
    closingCoef = 5
    whiteTopCoef = 5
    dilationCoedf = 5

    l1 = 1-l1
    e = skimage.morphology.closing(l1, skimage.morphology.disk(closingCoef))
    e2 = skimage.morphology.white_tophat(e,skimage.morphology.disk(whiteTopCoef))
    e2 = skimage.morphology.dilation(e2,skimage.morphology.disk(dilationCoedf))

    tmp = e.astype("int")-e2.astype("int")
    tmp[tmp<0] = 0
    tmp = tmp.astype("uint8")

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

    # draw contours on the original image
    image_copy = np.zeros_like(frame.copy())

    prev = 0
    max_radius = 0

    big_idx = -1
    max_x = (x_start[0]+y_start[0])/2
    max_y = (x_start[1]+y_start[1])/2

    w_max = np.abs(x_start[0]+y_start[0])
    h_max = np.abs(x_start[1]+y_start[1])

    for idx, i in enumerate(contours):

        current = cv2.contourArea(i)
        (x,y,w,h) = cv2.boundingRect(i)
        (x_probe ,y_probe ),radius_probe = cv2.minEnclosingCircle(i)

        border = (x_probe < y_start[0]-10) *(y_probe < y_start[1]-10 )*(x_probe > x_start[0]+10)*(y_probe > x_start[1]+10)
        area_cond_min =  (current > 8e3) 
        area_cond_max = (current < 2.5e6)

        x_borders =(np.sum(contours[idx][:,0,0] == 0) < th_num )*(np.sum(contours[idx][:,0,0] >= img.shape[0]-1) < th_num)
        y_borders =(np.sum(contours[idx][:,0,1] == 0) < th_num )*(np.sum(contours[idx][:,0,1] >= img.shape[0]-1) < th_num)
        
        if (area_cond_min) & (area_cond_max) & (prev < current) &  (border) & (x_borders*y_borders):
            
            big_idx = idx

            #Area and radius
            prev = current
            max_radius = radius

            #center
            max_x = x_
            max_y = y_

            #dimensions
            w_max = w
            h_max = h


    if big_idx == -1:
        start_pos = x_start
        end_pos = y_start
        area = -1;  x_ = -1; y_ = -1; radius = -1
    else:
        cv2.fillPoly(image_copy, pts = [contours[big_idx]], color=(2**16,0,0))
        (x_,y_),radius = cv2.minEnclosingCircle(contours[big_idx])
        area = cv2.contourArea(contours[big_idx])

        s_1 = int(x_-radius*1.5)
        if s_1<0:
            s_1 = 0

        e_1 = int(x_+radius*1.5)
        if e_1>2304:
            e_1=2304

        s_2 = int(y_+radius*1.5)
        if s_2 >2304:
            s_2 = 2304

        e_2 = int(y_-radius*1.5)
        if e_2 < 0:
            e_2 = 0


        start_pos = (s_1,e_2)
        end_pos = (e_1,s_2)

    return start_pos, end_pos, max_x, max_y, max_radius, prev, contours, big_idx

def check_box(x, y, r):

    s_1 = int(x-r*1.5)
    if s_1<0:
        s_1 = 0

    e_1 = int(x+r*1.5)
    if e_1>2304:
        e_1=2304

    s_2 = int(y+r*1.5)
    if s_2 >2304:
        s_2 = 2304

    e_2 = int(y-r*1.5)
    if e_2 < 0:
        e_2 = 0


    start_pos = (s_1,e_2)
    end_pos = (e_1,s_2)

    return start_pos, end_pos

def check_contour(i, prev, x_start, y_start, max_size):

    th_num = 500

    current = cv2.contourArea(i)
    (x,y,w,h) = cv2.boundingRect(i)
    (x_probe ,y_probe ), radius_probe = cv2.minEnclosingCircle(i)

    border = (x_probe < y_start[0]-10) *(y_probe < y_start[1]-10 )*(x_probe > x_start[0]+10)*(y_probe > x_start[1]+10)
    area_cond_min =  (current > 8e3) 
    area_cond_max = (current < 2.5e6)

    x_borders =(np.sum(i[:,0,0] == 0) < th_num )*(np.sum(i[:,0,0] >= max_size-1) < th_num)
    y_borders =(np.sum(i[:,0,1] == 0) < th_num )*(np.sum(i[:,0,1] >= max_size-1) < th_num)

    return ((area_cond_min) & (area_cond_max) & (prev < current) &  (border) & (x_borders*y_borders))

#Segmentation
def process_frame(img, x_start, y_start):

    frame = img.copy()
    th_num = 500

    frame = scipy.ndimage.gaussian_filter(frame, (5,5))
    frame = frame.astype(int)

    glob_thresh = threshold_otsu(frame)
    binary_local = frame > glob_thresh

    radius = 30   
    footprint = morphology.disk(radius)
    local_otsu = rank.otsu(frame.copy(), footprint)
    lo = frame >= local_otsu    
    l1 = np.zeros_like(lo)
    l1[binary_local] = lo[binary_local]

    # could/should be improved    
    closingCoef = 5
    whiteTopCoef = 5
    dilationCoedf = 5

    l1 = 1-l1
    e = skimage.morphology.closing(l1, skimage.morphology.disk(closingCoef))
    e2 = skimage.morphology.white_tophat(e,skimage.morphology.disk(whiteTopCoef))
    e2 = skimage.morphology.dilation(e2,skimage.morphology.disk(dilationCoedf))
    tmp = e.astype("int")-e2.astype("int")
    tmp[tmp<0] = 0
    tmp = tmp.astype("uint8")

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=tmp, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 

    # draw contours on the original image
    image_copy = np.zeros_like(frame.copy())

    prev = 0
    big_idx = -1

    for idx, i in enumerate(contours):

        current = cv2.contourArea(i)
        
        (x_probe ,y_probe ),radius_probe = cv2.minEnclosingCircle(i)
        
        border = (x_probe < y_start[0]-10) *(y_probe < y_start[1]-10 )*(x_probe > x_start[0]+10)*(y_probe > x_start[1]+10)
        
        area_cond_min =  (current > 8e3) 
        area_cond_max = (current < 2.5e6)

        x_borders =(np.sum(contours[idx][:,0,0] == 0) < th_num )*(np.sum(contours[idx][:,0,0] >= img.shape[0]-1) < th_num)
        y_borders =(np.sum(contours[idx][:,0,1] == 0) < th_num )*(np.sum(contours[idx][:,0,1] >= img.shape[0]-1) < th_num)
        
        if (area_cond_min) & (area_cond_max) & (prev < current) &  (border)  & (x_borders*y_borders):
            
            big_idx = idx
            prev = current


    if big_idx == -1:
        start_pos = x_start
        end_pos = y_start
        area = -1;  x_ = -1; y_ = -1; radius = -1
    else:
        cv2.fillPoly(image_copy, pts = [contours[big_idx]], color=(2**16,0,0))
        (x_,y_),radius = cv2.minEnclosingCircle(contours[big_idx])
        area = cv2.contourArea(contours[big_idx])

        s_1 = int(x_-radius*1.5)
        if s_1<0:
            s_1 = 0

        e_1 = int(x_+radius*1.5)
        if e_1>2304:
            e_1=2304

        s_2 = int(y_+radius*1.5)
        if s_2 >2304:
            s_2 = 2304

        e_2 = int(y_-radius*1.5)
        if e_2 < 0:
            e_2 = 0


        start_pos = (s_1,e_2)
        end_pos = (e_1,s_2)

    #cv2.rectangle(img_sec, start_pos, end_pos, (2**16,0,0), 5)

    #plt.imshow(np.hstack([e, e2, tmp]))
    #plt.show()

    #plt.imshow(np.hstack([img_sec, image_copy]))
    #plt.show()


    return x_, y_, radius, area, start_pos, end_pos, image_copy

#Concatenate dictionaries
def pile_data(current, total_dict, round, color):

    name = "loc_{}_ch_{}".format(round, color)
    total_dict[name] = {}

    total_dict[name]["x"] = []
    total_dict[name]["y"] = []
    total_dict[name]["z"] = []
    total_dict[name]["r"] = []
    total_dict[name]["area"] = []
    total_dict[name]["mask"] = []
    total_dict[name]["big_idx"] = []

    for i in range(len(current)):
        total_dict[name]["x"].append(current[i][0])
        total_dict[name]["y"].append(current[i][1])
        total_dict[name]["r"].append(current[i][2])
        total_dict[name]["area"].append(current[i][3])
        total_dict[name]["z"].append(current[i][4]) 
        total_dict[name]["mask"].append(current[i][5])
        total_dict[name]["big_idx"].append(current[i][6])

    return total_dict



import re
import xmltodict
import six
import numpy as np
import warnings

from nd2reader.common import read_chunk, read_array, read_metadata, parse_date, get_from_dict_if_exists
from nd2reader.common_raw_metadata import parse_dimension_text_line, parse_if_not_none, parse_roi_shape, parse_roi_type, get_loops_from_data, determine_sampling_interval


class RawMetadata(object):
    """RawMetadata class parses and stores the raw metadata that is read from the binary file in dict format.
    """

    def __init__(self, fh, label_map):
        self._fh = fh
        self._label_map = label_map
        self._metadata_parsed = None

    @property
    def __dict__(self):
        """Returns the parsed metadata in dictionary form.

        Returns:
            dict: the parsed metadata
        """
        return self.get_parsed_metadata()

    def get_parsed_metadata(self):
        """Returns the parsed metadata in dictionary form.

        Returns:
            dict: the parsed metadata
        """

        if self._metadata_parsed is not None:
            return self._metadata_parsed

        frames_per_channel = self._parse_total_images_per_channel()
        self._metadata_parsed = {
            "height": parse_if_not_none(self.image_attributes, self._parse_height),
            "width": parse_if_not_none(self.image_attributes, self._parse_width),
            "date": parse_if_not_none(self.image_text_info, self._parse_date),
            "fields_of_view": self._parse_fields_of_view(),
            "frames": self._parse_frames(),
            "z_levels": self._parse_z_levels(),
            "z_coordinates": parse_if_not_none(self.z_data, self._parse_z_coordinates),
            "total_images_per_channel": frames_per_channel,
            "channels": self._parse_channels(),
            "pixel_microns": parse_if_not_none(self.image_calibration, self._parse_calibration)
        }

        self._set_default_if_not_empty('fields_of_view')
        self._set_default_if_not_empty('frames')
        self._metadata_parsed['num_frames'] = len(self._metadata_parsed['frames'])

        self._parse_roi_metadata()
        self._parse_experiment_metadata()
        self._parse_events()

        return self._metadata_parsed

    def _set_default_if_not_empty(self, entry):
        total_images = self._metadata_parsed['total_images_per_channel'] \
            if self._metadata_parsed['total_images_per_channel'] is not None else 0

        if len(self._metadata_parsed[entry]) == 0 and total_images > 0:
            # if the file is not empty, we always have one of this entry
            self._metadata_parsed[entry] = [0]

    def _parse_width_or_height(self, key):
        try:
            length = self.image_attributes[six.b('SLxImageAttributes')][six.b(key)]
        except KeyError:
            length = None

        return length

    def _parse_height(self):
        return self._parse_width_or_height('uiHeight')

    def _parse_width(self):
        return self._parse_width_or_height('uiWidth')

    def _parse_date(self):
        try:
            return parse_date(self.image_text_info[six.b('SLxImageTextInfo')])
        except KeyError:
            return None

    def _parse_calibration(self):
        try:
            return self.image_calibration.get(six.b('SLxCalibration'), {}).get(six.b('dCalibration'))
        except KeyError:
            return None

    def _parse_frames(self):
        """The number of cycles.

        Returns:
            list: list of all the frame numbers
        """
        return self._parse_dimension(r""".*?T'?\((\d+)\).*?""")

    def _parse_channels(self):
        """These are labels created by the NIS Elements user. Typically they may a short description of the filter cube
        used (e.g. 'bright field', 'GFP', etc.)

        Returns:
            list: the color channels
        """
        if self.image_metadata_sequence is None:
            return []

        try:
            metadata = self.image_metadata_sequence[six.b('SLxPictureMetadata')][six.b('sPicturePlanes')]
        except KeyError:
            return []

        channels = self._process_channels_metadata(metadata)

        return channels

    def _process_channels_metadata(self, metadata):
        validity = self._get_channel_validity_list(metadata)

        # Channel information is contained in dictionaries with the keys a0, a1...an where the number
        # indicates the order in which the channel is stored. So by sorting the dicts alphabetically
        # we get the correct order.
        channels = []
        for valid, (label, chan) in zip(validity, sorted(metadata[six.b('sPlaneNew')].items())):
            if not valid:
                continue
            if chan[six.b('sDescription')] is not None:
                channels.append(chan[six.b('sDescription')].decode("utf8"))
            else:
                channels.append('Unknown')
        return channels

    def _get_channel_validity_list(self, metadata):
        try:
            validity = self.image_metadata[six.b('SLxExperiment')][six.b('ppNextLevelEx')][six.b('')][0][
                six.b('ppNextLevelEx')][six.b('')][0][six.b('pItemValid')]
        except (KeyError, TypeError):
            # If none of the channels have been deleted, there is no validity list, so we just make one
            validity = [True for _ in metadata]
        return validity

    def _parse_fields_of_view(self):
        """The metadata contains information about fields of view, but it contains it even if some fields
        of view were cropped. We can't find anything that states which fields of view are actually
        in the image data, so we have to calculate it. There probably is something somewhere, since
        NIS Elements can figure it out, but we haven't found it yet.

        """
        return self._parse_dimension(r""".*?XY\((\d+)\).*?""")

    def _parse_z_levels(self):
        """The different levels in the Z-plane.

        If they are not available from the _parse_dimension function AND there
        is NO 'Dimensions: ' textinfo item in the file, we return a range with
        the length of z_coordinates if available, otherwise an empty list.

        Returns:
            list: the z levels, just a sequence from 0 to n.
        """
        # get the dimension text to check if we should apply the fallback or not
        dimension_text = self._parse_dimension_text()

        # this returns range(len(z_levels))
        z_levels = self._parse_dimension(r""".*?Z\((\d+)\).*?""", dimension_text)

        if len(z_levels) > 0 or len(dimension_text) > 0:
            # Either we have found the z_levels (first condition) so return, or
            # don't fallback, because Z is apparently not in Dimensions, so
            # there should be no z_levels
            return z_levels

        # Not available from dimension, get from z_coordinates
        z_levels = parse_if_not_none(self.z_data, self._parse_z_coordinates)

        if z_levels is None:
            # No z coordinates, return empty list
            return []

        warnings.warn("Z-levels details missing in metadata. Using Z-coordinates instead.")
        return range(len(z_levels))

    def _parse_z_coordinates(self):
        """The coordinate in micron for all z planes.

        Returns:
            list: the z coordinates in micron
        """
        return self.z_data.tolist()

    def _parse_dimension_text(self):
        """While there are metadata values that represent a lot of what we want to capture, they seem to be unreliable.
        Sometimes certain elements don't exist, or change their data type randomly. However, the human-readable text
        is always there and in the same exact format, so we just parse that instead.

        """
        dimension_text = six.b("")
        if self.image_text_info is None:
            return dimension_text

        try:
            textinfo = self.image_text_info[six.b('SLxImageTextInfo')].values()
        except KeyError:
            return dimension_text

        for line in textinfo:
            entry = parse_dimension_text_line(line)
            if entry is not None:
                return entry

        return dimension_text

    def _parse_dimension(self, pattern, dimension_text=None):
        dimension_text = self._parse_dimension_text() if dimension_text is None else dimension_text
        if dimension_text is None:
            return []

        if six.PY3:
            dimension_text = dimension_text.decode("utf8")

        match = re.match(pattern, dimension_text)
        if not match:
            return []

        count = int(match.group(1))
        return range(count)

    def _parse_total_images_per_channel(self):
        """The total number of images per channel.

        Warning: this may be inaccurate as it includes 'gap' images.

        """
        if self.image_attributes is None:
            return 0
        try:
            total_images = self.image_attributes[six.b('SLxImageAttributes')][six.b('uiSequenceCount')]
        except KeyError:
            total_images = None

        return total_images

    def _parse_roi_metadata(self):
        """Parse the raw ROI metadata.

        """
        if self.roi_metadata is None or not six.b('RoiMetadata_v1') in self.roi_metadata:
            return

        raw_roi_data = self.roi_metadata[six.b('RoiMetadata_v1')]

        if not six.b('m_vectGlobal_Size') in raw_roi_data:
            return

        number_of_rois = raw_roi_data[six.b('m_vectGlobal_Size')]

        roi_objects = []
        for i in range(number_of_rois):
            current_roi = raw_roi_data[six.b('m_vectGlobal_%d' % i)]
            roi_objects.append(self._parse_roi(current_roi))

        self._metadata_parsed['rois'] = roi_objects

    def _parse_roi(self, raw_roi_dict):
        """Extract the vector animation parameters from the ROI.

        This includes the position and size at the given timepoints.

        Args:
            raw_roi_dict: dictionary of raw roi metadata

        Returns:
            dict: the parsed ROI metadata

        """
        number_of_timepoints = raw_roi_dict[six.b('m_vectAnimParams_Size')]

        roi_dict = {
            "timepoints": [],
            "positions": [],
            "sizes": [],
            "shape": parse_roi_shape(raw_roi_dict[six.b('m_sInfo')][six.b('m_uiShapeType')]),
            "type": parse_roi_type(raw_roi_dict[six.b('m_sInfo')][six.b('m_uiInterpType')])
        }
        for i in range(number_of_timepoints):
            roi_dict = self._parse_vect_anim(roi_dict, raw_roi_dict[six.b('m_vectAnimParams_%d' % i)])

        # convert to NumPy arrays
        roi_dict["timepoints"] = np.array(roi_dict["timepoints"], dtype=np.float)
        roi_dict["positions"] = np.array(roi_dict["positions"], dtype=np.float)
        roi_dict["sizes"] = np.array(roi_dict["sizes"], dtype=np.float)

        return roi_dict

    def _parse_vect_anim(self, roi_dict, animation_dict):
        """
        Parses a ROI vector animation object and adds it to the global list of timepoints and positions.

        Args:
            roi_dict: the raw roi dictionary
            animation_dict: the raw animation dictionary

        Returns:
            dict: the parsed metadata

        """
        roi_dict["timepoints"].append(animation_dict[six.b('m_dTimeMs')])

        image_width = self._metadata_parsed["width"] * self._metadata_parsed["pixel_microns"]
        image_height = self._metadata_parsed["height"] * self._metadata_parsed["pixel_microns"]

        # positions are taken from the center of the image as a fraction of the half width/height of the image
        position = np.array((0.5 * image_width * (1 + animation_dict[six.b('m_dCenterX')]),
                             0.5 * image_height * (1 + animation_dict[six.b('m_dCenterY')]),
                             animation_dict[six.b('m_dCenterZ')]))
        roi_dict["positions"].append(position)

        size_dict = animation_dict[six.b('m_sBoxShape')]

        # sizes are fractions of the half width/height of the image
        roi_dict["sizes"].append((size_dict[six.b('m_dSizeX')] * 0.25 * image_width,
                                  size_dict[six.b('m_dSizeY')] * 0.25 * image_height,
                                  size_dict[six.b('m_dSizeZ')]))
        return roi_dict

    def _parse_experiment_metadata(self):
        """Parse the metadata of the ND experiment

        """
        self._metadata_parsed['experiment'] = {
            'description': 'unknown',
            'loops': []
        }

        if self.image_metadata is None or six.b('SLxExperiment') not in self.image_metadata:
            return

        raw_data = self.image_metadata[six.b('SLxExperiment')]

        if six.b('wsApplicationDesc') in raw_data:
            self._metadata_parsed['experiment']['description'] = raw_data[six.b('wsApplicationDesc')].decode('utf8')

        if six.b('uLoopPars') in raw_data:
            self._metadata_parsed['experiment']['loops'] = self._parse_loop_data(raw_data[six.b('uLoopPars')])

    def _parse_loop_data(self, loop_data):
        """Parse the experimental loop data

        Args:
            loop_data: dictionary of experiment loops

        Returns:
            list: list of the parsed loops

        """
        loops = get_loops_from_data(loop_data)

        # take into account the absolute time in ms
        time_offset = 0

        parsed_loops = []

        for loop in loops:
            # duration of this loop
            duration = get_from_dict_if_exists('dDuration', loop) or 0
            interval = determine_sampling_interval(duration, loop)

            # if duration is not saved, infer it
            duration = self.get_duration_from_interval_and_loops(duration, interval, loop)

            # uiLoopType == 6 is a stimulation loop
            is_stimulation = get_from_dict_if_exists('uiLoopType', loop) == 6

            parsed_loop = {
                'start': time_offset,
                'duration': duration,
                'stimulation': is_stimulation,
                'sampling_interval': interval
            }

            parsed_loops.append(parsed_loop)

            # increase the time offset
            time_offset += duration

        return parsed_loops

    def get_duration_from_interval_and_loops(self, duration, interval, loop):
        """Infers the duration of the loop from the number of measurements and the interval

        Args:
            duration: loop duration in milliseconds
            duration: measurement interval in milliseconds
            loop: loop dictionary

        Returns:
            float: the loop duration in milliseconds

        """
        if duration == 0 and interval > 0:
            number_of_loops = get_from_dict_if_exists('uiCount', loop)
            number_of_loops = number_of_loops if number_of_loops is not None and number_of_loops > 0 else 1
            duration = interval * number_of_loops

        return duration

    def _parse_events(self):
        """Extract events

        """

        # list of event names manually extracted from an ND2 file that contains all manually
        # insertable events from NIS-Elements software (4.60.00 (Build 1171) Patch 02)
        event_names = {
            1: 'Autofocus',
            7: 'Command Executed',
            9: 'Experiment Paused',
            10: 'Experiment Resumed',
            11: 'Experiment Stopped by User',
            13: 'Next Phase Moved by User',
            14: 'Experiment Paused for Refocusing',
            16: 'External Stimulation',
            33: 'User 1',
            34: 'User 2',
            35: 'User 3',
            36: 'User 4',
            37: 'User 5',
            38: 'User 6',
            39: 'User 7',
            40: 'User 8',
            44: 'No Acquisition Phase Start',
            45: 'No Acquisition Phase End',
            46: 'Hardware Error',
            47: 'N-STORM',
            48: 'Incubation Info',
            49: 'Incubation Error'
        }

        self._metadata_parsed['events'] = []

        events = read_metadata(read_chunk(self._fh, self._label_map.image_events), 1)

        if events is None or six.b('RLxExperimentRecord') not in events:
            return

        events = events[six.b('RLxExperimentRecord')][six.b('pEvents')]

        if len(events) == 0:
            return

        for event in events[six.b('')]:
            event_info = {
                'index': event[six.b('I')],
                'time': event[six.b('T')],
                'type': event[six.b('M')],
            }
            if event_info['type'] in event_names.keys():
                event_info['name'] = event_names[event_info['type']]

            self._metadata_parsed['events'].append(event_info)

    @property
    def image_text_info(self):
        """Textual image information

        Returns:
            dict: containing the textual image info

        """
        return read_metadata(read_chunk(self._fh, self._label_map.image_text_info), 1)

    @property
    def image_metadata_sequence(self):
        """Image metadata of the sequence

        Returns:
            dict: containing the metadata

        """
        return read_metadata(read_chunk(self._fh, self._label_map.image_metadata_sequence), 1)

    @property
    def image_calibration(self):
        """The amount of pixels per micron.

        Returns:
            dict: pixels per micron
        """
        return read_metadata(read_chunk(self._fh, self._label_map.image_calibration), 1)

    @property
    def image_attributes(self):
        """Image attributes

        Returns:
            dict: containing the image attributes
        """
        return read_metadata(read_chunk(self._fh, self._label_map.image_attributes), 1)

    @property
    def x_data(self):
        """X data

        Returns:
            dict: x_data
        """
        return read_array(self._fh, 'double', self._label_map.x_data)

    @property
    def y_data(self):
        """Y data

        Returns:
            dict: y_data
        """
        return read_array(self._fh, 'double', self._label_map.y_data)

    @property
    def z_data(self):
        """Z data

        Returns:
            dict: z_data
        """
        try:
            return read_array(self._fh, 'double', self._label_map.z_data)
        except ValueError:
            # Depending on the file format/exact settings, this value is
            # sometimes saved as float instead of double
            return read_array(self._fh, 'float', self._label_map.z_data)

    @property
    def roi_metadata(self):
        """Contains information about the defined ROIs: shape, position and type (reference/background/stimulation).

        Returns:
            dict: ROI metadata dictionary
        """
        return read_metadata(read_chunk(self._fh, self._label_map.roi_metadata), 1)

    @property
    def pfs_status(self):
        """Perfect focus system (PFS) status

        Returns:
            dict: Perfect focus system (PFS) status

        """
        return read_array(self._fh, 'int', self._label_map.pfs_status)

    @property
    def pfs_offset(self):
        """Perfect focus system (PFS) offset

        Returns:
            dict: Perfect focus system (PFS) offset

        """
        return read_array(self._fh, 'int', self._label_map.pfs_offset)

    @property
    def camera_exposure_time(self):
        """Exposure time information

        Returns:
            dict: Camera exposure time

        """
        return read_array(self._fh, 'double', self._label_map.camera_exposure_time)

    @property
    def lut_data(self):
        """LUT information

        Returns:
            dict: LUT information

        """
        return xmltodict.parse(read_chunk(self._fh, self._label_map.lut_data))

    @property
    def grabber_settings(self):
        """Grabber settings

        Returns:
            dict: Acquisition settings

        """
        return xmltodict.parse(read_chunk(self._fh, self._label_map.grabber_settings))

    @property
    def custom_data(self):
        """Custom user data

        Returns:
            dict: custom user data

        """
        return xmltodict.parse(read_chunk(self._fh, self._label_map.custom_data))

    @property
    def app_info(self):
        """NIS elements application info

        Returns:
            dict: (Version) information of the NIS Elements application

        """
        return xmltodict.parse(read_chunk(self._fh, self._label_map.app_info))

    @property
    def camera_temp(self):
        """Camera temperature

        Yields:
            float: the temperature

        """
        camera_temp = read_array(self._fh, 'double', self._label_map.camera_temp)
        if camera_temp:
            for temp in map(lambda x: round(x * 100.0, 2), camera_temp):
                yield temp

    @property
    def acquisition_times(self):
        """Acquisition times

        Yields:
            float: the acquisition time

        """
        acquisition_times = read_array(self._fh, 'double', self._label_map.acquisition_times)
        if acquisition_times:
            for acquisition_time in map(lambda x: x / 1000.0, acquisition_times):
                yield acquisition_time

    @property
    def image_metadata(self):
        """Image metadata

        Returns:
            dict: Extra image metadata

        """
        if self._label_map.image_metadata:
            return read_metadata(read_chunk(self._fh, self._label_map.image_metadata), 1)

    @property
    def image_events(self):
        """Image events

        Returns:
            dict: Image events
        """
        if self._label_map.image_metadata:
            for event in self._metadata_parsed["events"]:
                yield event




class LabelMap(object):
    """Contains pointers to metadata. This might only be valid for V3 files.

    """

    def __init__(self, raw_binary_data):
        self._data = raw_binary_data
        self._image_data = {}

    def _get_location(self, label):
        try:
            label_location = self._data.index(label) + len(label)
            return self._parse_data_location(label_location)
        except ValueError:
            return None

    def _parse_data_location(self, label_location):
        location, length = struct.unpack("QQ", self._data[label_location: label_location + 16])
        return location

    @property
    def image_text_info(self):
        """Get the location of the textual image information

        Returns:
            int: The location of the textual image information

        """
        return self._get_location(six.b("ImageTextInfoLV!"))

    @property
    def image_metadata(self):
        """Get the location of the image metadata

        Returns:
            int: The location of the image metadata

        """
        return self._get_location(six.b("ImageMetadataLV!"))

    @property
    def image_events(self):
        """Get the location of the image events

        Returns:
            int: The location of the image events

        """
        return self._get_location(six.b("ImageEventsLV!"))

    @property
    def image_metadata_sequence(self):
        """Get the location of the image metadata sequence. There is always only one of these, even though it has a pipe
         followed by a zero, which is how they do indexes.

        Returns:
            int: The location of the image metadata sequence

        """
        return self._get_location(six.b("ImageMetadataSeqLV|0!"))

    def get_image_data_location(self, index):
        """Get the location of the image data

        Returns:
            int: The location of the image data

        """
        if not self._image_data:
            regex = re.compile(six.b("""ImageDataSeq\|(\d+)!"""))
            for match in regex.finditer(self._data):
                if match:
                    location = self._parse_data_location(match.end())
                    self._image_data[int(match.group(1))] = location
        return self._image_data[index]

    @property
    def image_calibration(self):
        """Get the location of the image calibration

        Returns:
            int: The location of the image calibration

        """
        return self._get_location(six.b("ImageCalibrationLV|0!"))

    @property
    def image_attributes(self):
        """Get the location of the image attributes

        Returns:
            int: The location of the image attributes

        """
        return self._get_location(six.b("ImageAttributesLV!"))

    @property
    def x_data(self):
        """Get the location of the custom x data

        Returns:
            int: The location of the custom x data

        """
        return self._get_location(six.b("CustomData|X!"))

    @property
    def y_data(self):
        """Get the location of the custom y data

        Returns:
            int: The location of the custom y data

        """
        return self._get_location(six.b("CustomData|Y!"))

    @property
    def z_data(self):
        """Get the location of the custom z data

        Returns:
            int: The location of the custom z data

        """
        return self._get_location(six.b("CustomData|Z!"))

    @property
    def roi_metadata(self):
        """Information about any regions of interest (ROIs) defined in the nd2 file

        Returns:
            int: The location of the regions of interest (ROIs)

        """
        return self._get_location(six.b("CustomData|RoiMetadata_v1!"))

    @property
    def pfs_status(self):
        """Get the location of the perfect focus system (PFS) status

        Returns:
            int: The location of the perfect focus system (PFS) status

        """
        return self._get_location(six.b("CustomData|PFS_STATUS!"))

    @property
    def pfs_offset(self):
        """Get the location of the perfect focus system (PFS) offset

        Returns:
            int: The location of the perfect focus system (PFS) offset

        """
        return self._get_location(six.b("CustomData|PFS_OFFSET!"))

    @property
    def guid(self):
        """Get the location of the image guid

        Returns:
            int: The location of the image guid

        """
        return self._get_location(six.b("CustomData|GUIDStore!"))

    @property
    def description(self):
        """Get the location of the image description

        Returns:
            int: The location of the image description

        """
        return self._get_location(six.b("CustomData|CustomDescriptionV1_0!"))

    @property
    def camera_exposure_time(self):
        """Get the location of the camera exposure time

        Returns:
            int: The location of the camera exposure time

        """
        return self._get_location(six.b("CustomData|Camera_ExposureTime1!"))

    @property
    def camera_temp(self):
        """Get the location of the camera temperature

        Returns:
            int: The location of the camera temperature

        """
        return self._get_location(six.b("CustomData|CameraTemp1!"))

    @property
    def acquisition_times(self):
        """Get the location of the acquisition times, block 1

        Returns:
            int: The location of the acquisition times, block 1

        """
        return self._get_location(six.b("CustomData|AcqTimesCache!"))

    @property
    def acquisition_times_2(self):
        """Get the location of the acquisition times, block 2

        Returns:
            int: The location of the acquisition times, block 2

        """
        return self._get_location(six.b("CustomData|AcqTimes2Cache!"))

    @property
    def acquisition_frames(self):
        """Get the location of the acquisition frames

        Returns:
            int: The location of the acquisition frames

        """
        return self._get_location(six.b("CustomData|AcqFramesCache!"))

    @property
    def lut_data(self):
        """Get the location of the LUT data

        Returns:
            int: The location of the LUT data

        """
        return self._get_location(six.b("CustomDataVar|LUTDataV1_0!"))

    @property
    def grabber_settings(self):
        """Get the location of the grabber settings

        Returns:
            int: The location of the grabber settings

        """
        return self._get_location(six.b("CustomDataVar|GrabberCameraSettingsV1_0!"))

    @property
    def custom_data(self):
        """Get the location of the custom user data

        Returns:
            int: The location of the custom user data

        """
        return self._get_location(six.b("CustomDataVar|CustomDataV2_0!"))

    @property
    def app_info(self):
        """Get the location of the application info metadata

        Returns:
            int: The location of the application info metadata

        """
        return self._get_location(six.b("CustomDataVar|AppInfo_V1_0!"))
    


def calc_MSD_theta_tau(data, tau, dt):

    x = data["x"] - data["x"][0]
    y = data["y"] - data["y"][0]

    dx = np.gradient(x)
    dy = np.gradient(y)

    d_theta = np.zeros(len(x))
    idx_theta = 1 

    MSD = np.zeros(len(x))
    ACF = np.zeros(len(x))

    velocity_tau = np.zeros((len(x),2))
    velocity_2tau = np.zeros((len(x),2))


    for idx in np.arange(len(x)):

        if idx+2*tau >= len(x):

            if idx+tau >= len(x):
                break
            velocity_tau[idx,0] = (x[idx+tau]-x[idx])/(tau*dt)
            velocity_tau[idx,1] = (y[idx+tau]-y[idx])/(tau*dt)

            ACF[idx] = dx[idx]*dx[idx+tau] + dy[idx]*dy[idx+tau]
            MSD[idx]= np.sqrt( ((x[idx+tau]-x[idx])/dt)**2 + ((y[idx+tau]-y[idx])/dt)**2)
            
        else:
            velocity_tau[idx,0] = (x[idx+tau]-x[idx])/(tau*dt)
            velocity_tau[idx,1] = (y[idx+tau]-y[idx])/(tau*dt)

            velocity_2tau[idx,0] = (x[idx+2*tau]-x[idx])/(2*(tau*dt))
            velocity_2tau[idx,1] = (y[idx+2*tau]-y[idx])/(2*(tau*dt))

            d_theta[idx] = np.arccos(np.sum(velocity_tau[idx,:]*velocity_2tau[idx,:])/(np.sqrt(velocity_tau[idx,0]**2 + velocity_tau[idx,1]**2 )*np.sqrt(velocity_2tau[idx,0]**2 + velocity_2tau[idx,1]**2 )))

            idx_theta = idx
    
    return d_theta[:idx_theta].mean(), velocity_tau[:idx,:], MSD[:idx].mean(), ACF[:idx].mean(), idx_theta, idx, d_theta[:idx_theta]


def calc_SVD(x, y, dt): 

    velocity = np.zeros((len(x)-1,2))

    for idx in np.arange(len(x)-1):
        velocity[idx,0] = (x[idx+1]-x[idx])/dt
        velocity[idx,1] = (y[idx+1]-y[idx])/dt

    U, S, Vh = np.linalg.svd(velocity, full_matrices=True)

    return velocity, U, S, Vh

def calc_MSD_theta(data, dt, tau = 1):

    x = data["x"] - data["x"][0]
    y = data["y"] - data["y"][0]

    d_theta = np.zeros(len(x))
    idx_theta = 1 

    MSD = np.zeros(len(x))

    velocity_tau = np.zeros((len(x),2))
    velocity_2tau = np.zeros((len(x),2))


    for idx in np.arange(len(x)):

        if idx+2*tau >= len(x):

            if idx+tau >= len(x):
                break

            velocity_tau[idx,0] = (x[idx+tau]-x[idx])/(tau*dt)
            velocity_tau[idx,1] = (y[idx+tau]-y[idx])/(tau*dt)

            MSD[idx]= np.sqrt( ((x[idx+tau]-x[idx])/dt)**2 + ((y[idx+tau]-y[idx])/dt)**2)
            
        else:
            velocity_tau[idx,0] = (x[idx+tau]-x[idx])/(tau*dt)
            velocity_tau[idx,1] = (y[idx+tau]-y[idx])/(tau*dt)

            velocity_2tau[idx,0] = (x[idx+2*tau]-x[idx])/(2*(tau*dt))
            velocity_2tau[idx,1] = (y[idx+2*tau]-y[idx])/(2*(tau*dt))

            d_theta[idx] = np.arccos(np.sum(velocity_tau[idx,:]*velocity_2tau[idx,:])/(np.sqrt(velocity_tau[idx,0]**2 + velocity_tau[idx,1]**2 )*np.sqrt(velocity_2tau[idx,0]**2 + velocity_2tau[idx,1]**2 )))

            idx_theta = idx
    
    return d_theta[:idx_theta], velocity_tau[:idx,:], MSD[:idx], idx_theta, idx

def power_law(x, a, k):
    return a*x**(k)

def calc_MSD(data, dt, tau):
    
    x = data["x"] - data["x"].values[0]
    y = data["y"] - data["y"].values[0]

    MSD = np.zeros(len(x))

    for idx in np.arange(len(x)):
        if idx+tau >= len(x):
            break

        MSD[idx]= np.sqrt( ((x[idx+tau]-x[idx])/dt)**2 + ((y[idx+tau]-y[idx])/dt)**2)
            
    
    return MSD[:idx].mean()


"""

Stored frame by frame analysis

loc_id = 0
t_id = 0

out_name = os.path.join(results,'{}_{}_{}.mp4'.format(os.path.split(video_path)[1][:-4], (loc_id), datetime.date.today() ) )

out_process = ( 
ffmpeg 
.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'
.format(2304, 2304)) 
.output(out_name, pix_fmt='yuv420p') .overwrite_output() 
.run_async(pipe_stdin=True) 
)

channel = "BF"

track_list = []
total_dict = {}

with ND2Reader(video_path) as images:
    #c -channel, v - locations, t - time
    
    metas = load_metadata(images)
    images.iter_axes = "vt"
    images.bundle_axes = "zyx"


    vis_level = int(metas["n_levels"]/2)
    frames_tot = int((metas["n_channels"]*metas["n_fields"]*metas["n_frames"]))

    
    x_start = coords[loc_id][0][0] #(0,2304)
    y_start = coords[loc_id][0][1] #(2304,0) 


    for idx, img in tqdm.tqdm(enumerate(images),total=frames_tot):

        c_id = idx%metas["n_channels"]
        id_name = metas["channels"][c_id]


        if (id_name != channel) : #| (loc_id != 1)
            #t_id += 1
            #if t_id == metas["n_frames"]:
            #    t_id = 0
            #    loc_id += 1
            continue

        focus_level = 1e10
        focus_idx = 0

        x_final = x_start
        y_final = y_start

        for i in range(img.shape[0]):
    
            current_frame = img[i].copy()

            th = np.percentile(current_frame.flatten(),5)
            current_frame[current_frame>th] = 0
            
            x, y, r, area, x_start_, y_start_, mask = process_frame(current_frame, x_start, y_start)
            if area == -1:
                continue
            else:
                cropped = current_frame[x_start_[1]:y_start_[1], x_start_[0]:y_start_[0]]
                #cv2.rectangle(img_normalized, x_start_, y_start_, (2**16,0,0), 5)
                #cv2.imshow("win",cv2.resize(img_normalized, (512,512)))
                #k = cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                if (cropped.shape[0] == 0) or (cropped.shape[1] == 0):
                    lap = np.inf
                else:
                    lap = cv2.Laplacian(cropped, cv2.CV_64F).var()

                if focus_level > lap:

                    focus_level = lap
                    focus_idx = i

                    x_final = x_start_
                    y_final = y_start_
           
        x, y, r, area, x_start, y_start, mask = process_frame(img[focus_idx], x_final, y_final)

        int_frame = (img[focus_idx]*2**8/(2**16)).astype("uint8")
        int_frame = np.stack((int_frame,int_frame,int_frame), axis = -1)
        img_normalized = cv2.normalize(int_frame, None, 0, 255, cv2.NORM_MINMAX)
        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img_normalized, x_start, y_start, (2**16,0,0), 5)

        #int_frame = increase_brightness(int_frame, 100)
        int_mask = (mask.astype(bool)*255).astype("uint8")
        int_mask = np.stack((int_mask,int_mask,int_mask), axis = -1)
        
        out_vis = cv2.addWeighted(img_normalized,1.0,int_mask,0.75,0)
        
        #cv2.imshow("win",cv2.resize(out_vis, (512,512)))
        #k = cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #if k == "q":
        #    break
        plt.imshow(out_vis)
        plt.show()

        out_process.stdin.write(out_vis)

        track_list.append([x*metas["m"], y*metas["m"], r*metas["m"], area*metas["m"]**2, (focus_idx-1)*metas["z_step"], mask])


         
        t_id += 1
        if t_id == metas["n_frames"]:
            total_dict = pile_data(track_list, total_dict, loc_id, channel)

            out_process.stdin.close()
            out_process.wait()

            t_id = 0
            loc_id += 1
            track_list = []
            if loc_id == metas["n_fields"]:
                break
            
            #if loc_id == 2:
            #    break

            x_start = coords[loc_id][0][0] #(0,2304)
            y_start = coords[loc_id][0][1] #(2304,0) 
            out_name = os.path.join(results,'{}_{}_{}.mp4'.format(os.path.split(video_path)[1][:-4], (loc_id), datetime.date.today() ) )
            out_process = ( 
            ffmpeg 
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'
            .format(2304, 2304)) 
            .output(out_name, pix_fmt='yuv420p') .overwrite_output() 
            .run_async(pipe_stdin=True) 
            )


with open(os.path.join(results,'{}_detections.pkl'.format(os.path.split(video_path)[1][:-4])), 'wb') as f:
    pickle.dump(total_dict, f)


"""