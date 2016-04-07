# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

# def im_list_to_blob(ims):
#     """Convert a list of images into a network input.
# 
#     Assumes images are already prepared (means subtracted, BGR order, ...).
#     """
#     max_shape = np.array([im.shape for im in ims]).max(axis=0)
#     num_images = len(ims)
#     blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
#                     dtype=np.float32)
#     for i in xrange(num_images):
#         im = ims[i]
#         blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
#     # Move channels (axis 3) to axis 1
#     # Axis order will become: (batch elem, channel, height, width)
#     channel_swap = (0, 3, 1, 2)
#     blob = blob.transpose(channel_swap)
#     return blob

def im_list_to_blob(ims, dsms=None):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    num_dsms = 0
    if dsms is not None:
        num_dsms = len(dsms)
    has_dsm = False
    if num_dsms == num_images:
        has_dsm = True
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), 
                    dtype=np.float32)
    dsm_blob = None 
    if has_dsm:
        dsm_blob = np.zeros((num_images, max_shape[0], max_shape[1], 1), 
                            dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        if has_dsm:
            dsm = dsms[i]
            dsm_blob[i, 0:im.shape[0], 0:im.shape[1], 0] = dsm
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    if has_dsm:
        dsm_blob = dsm_blob.transpose(channel_swap)
    return blob, dsm_blob

def im_list_to_blob_with_seg(ims, dsms=None, segs=None):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    num_dsms = 0
    if dsms is not None:
        num_dsms = len(dsms)
    has_dsm = False
    if num_dsms == num_images:
        has_dsm = True
    num_segs = 0
    if segs is not None:
        num_segs = len(segs)
    has_seg = False
    if num_segs == num_images:
        has_seg = True
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), 
                    dtype=np.float32)
    dsm_blob = None 
    if has_dsm:
        dsm_blob = np.zeros((num_images, max_shape[0], max_shape[1], 1), 
                            dtype=np.float32)
    seg_blob = None
    if has_seg:
        seg_blob = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                            dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        if has_dsm:
            dsm = dsms[i]
            dsm_blob[i, 0:im.shape[0], 0:im.shape[1], 0] = dsm
        if has_seg:
            seg = segs[i]
            seg_blob[i, 0:im.shape[0], 0:im.shape[1], 0] = seg
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    if has_dsm:
        dsm_blob = dsm_blob.transpose(channel_swap)
    if has_seg:
        seg_blob = seg_blob.transpose(channel_swap)
    return blob, dsm_blob, seg_blob

# def prep_im_for_blob(im, pixel_means, target_size, max_size):
#     """Mean subtract and scale an image for use in a blob."""
#     im = im.astype(np.float32, copy=False)
#     im -= pixel_means
#     im_shape = im.shape
#     im_size_min = np.min(im_shape[0:2])
#     im_size_max = np.max(im_shape[0:2])
#     im_scale = float(target_size) / float(im_size_min)
#     # Prevent the biggest axis from being more than MAX_SIZE
#     if np.round(im_scale * im_size_max) > max_size:
#         im_scale = float(max_size) / float(im_size_max)
#     im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
#                     interpolation=cv2.INTER_LINEAR)
# 
#     return im, im_scale

def prep_im_for_blob(im, pixel_means, target_size, max_size, dsm=None, dsm_means=None):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means 
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    if dsm is not None:
        dsm = dsm.astype(np.float32, copy=False) 
        dsm -= dsm_means
        dsm = cv2.resize(dsm, None, None, fx=im_scale, fy=im_scale, 
                         interpolation=cv2.INTER_LINEAR)
        return im, dsm, im_scale
    return im, [], im_scale

def prep_im_for_blob_with_seg(im, pixel_means, target_size, max_size, dsm=None, dsm_means=None, seg=None):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means 
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    dsm1 = []
    if dsm is not None:
        dsm1 = dsm.astype(np.float32, copy=False) 
        dsm1 -= dsm_means
        dsm1 = cv2.resize(dsm, None, None, fx=im_scale, fy=im_scale, 
                         interpolation=cv2.INTER_LINEAR)
    seg1 = []
    if seg is not None:
        seg1 = seg.astype(np.float32, copy=False)
        seg1 = cv2.resize(seg, None, None, fx=im_scale, fy=im_scale, 
                         interpolation=cv2.INTER_NEAREST)
    return im, dsm1, seg1, im_scale