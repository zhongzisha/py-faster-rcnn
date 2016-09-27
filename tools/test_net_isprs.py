'''
To generate MATLAB result:
export PATH=/usr/local/MATLAB/R2015b/bin:$PATH
python ./tools/reval.py --imdb voc_2014_val --matlab --comp \
    ./output/faster_rcnn_end2end/voc_2014_val/vgg16_faster_rcnn_iter_70000/
    
    
Usage:
python tools/test_net_bgrd.py --gpu 2 \
--prototxt models/vaihingen/VGG16/faster_rcnn_end2end/test.prototxt \
--caffemodel output/faster_rcnn_end2end/voc_2013_train/vgg16_faster_rcnn_iter_70000.caffemodel \
--devkit_path /home/nlpr2/data/VOCdevkit2013 \
--year 2013 \
--image_set val \
--save_prefix vgg16 \
--nms_threshold 0.3 \
--set PIXEL_MEANS '[[[84,86,117]]]'

python tools/test_net_bgrd.py --gpu 3 \
--prototxt models/potsdam_d/VGG16/faster_rcnn_end2end/test.prototxt \
--caffemodel output/faster_rcnn_end2end/voc_2014_train/d_vgg16_faster_rcnn_iter_70000.caffemodel \
--devkit_path /home/nlpr2/data/VOCdevkit2014_for_test \
--year 2014 \
--image_set val \
--save_prefix d_vgg16 \
--nms_threshold 0.3 \
--set PIXEL_MEANS '[[[88,94,87]]]' DSM_MEANS '[42]' TEST.HAS_DSM True
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import caffe, os, sys, cv2
import argparse
from ast import literal_eval

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--prototxt', dest='prototxt', help='path of test.prototxt',
                        default=None, type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='path of caffemodel',
                        default=None, type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='path of devkit_path',
                        default=None, type=str)
    parser.add_argument('--year', dest='year', help='year',
                        default=None, type=str)
    parser.add_argument('--image_set', dest='image_set', help='image_set (train, val, test)',
                        default=None, type=str)
    parser.add_argument('--save_prefix', dest='save_prefix', help='save_prefix',
                        default=None, type=str)
    parser.add_argument('--step_size', dest='step_size', help='step_size',
                        default=250, type=int)
    parser.add_argument('--bgr_mean', dest='bgr_mean', help='bgr_mean',
                        default=None, type=str)
    parser.add_argument('--dsm_mean', dest='dsm_mean', help='dsm_mean',
                        default=None, type=str)
    parser.add_argument('--prob_blob_name', dest='prob_blob_name', help='prob_blob_name',
                        default='prob', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()

    return args

def get_seg_result(net, im, bgr_mean, dsm=None, dsm_mean=None, prob_blob_name='prob'):
    blobs = {'data' : None, 'dsm' : None}
    data_blob = np.zeros((1, im.shape[0], im.shape[1], im.shape[2]), 
                            dtype=np.float32)
    data_blob[0, :, :, :] = im - bgr_mean  
    channel_swap = (0, 3, 1, 2)
    blobs['data'] = data_blob.transpose(channel_swap)  
    net.blobs['data'].reshape(*(blobs['data'].shape))
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if dsm is not None:
        dsm_blob = np.zeros((1, im.shape[0], im.shape[1], 1), 
                                dtype=np.float32)
        dsm_blob[0, :, :, 0] = dsm - dsm_mean 
        blobs['dsm'] = dsm_blob.transpose(channel_swap)
        net.blobs['dsm'].reshape(*(blobs['dsm'].shape))
        forward_kwargs['dsm'] = blobs['dsm'].astype(np.float32, copy=False)
        
    blobs_out = net.forward(**forward_kwargs)
    
    seg_prob = net.blobs[prob_blob_name].data
    return seg_prob

def test_on_one_image(net, bgr0, bgr_mean, dsm0=None, dsm_mean=None, step_size=250, num_seg_classes=6, prob_blob_name='prob'):
    BLOCK_SIZE = 500  
        
    height, width, num_bands = bgr0.shape
    height_padsize = 0
    width_padsize = 0
    if height % BLOCK_SIZE:
        height_padsize = BLOCK_SIZE - height % BLOCK_SIZE
    if width % BLOCK_SIZE:
        width_padsize = BLOCK_SIZE - width % BLOCK_SIZE
    bgr0 = np.pad(bgr0, ((0, height_padsize),(0, width_padsize),(0, 0)), 'reflect')
    if dsm0 is not None:
        if len(dsm0.shape) == 2:
            dsm0 = np.pad(dsm0, ((0, height_padsize),(0, width_padsize)), 'reflect')
        else:
            dsm0 = np.pad(dsm0, ((0, 0), (0, height_padsize),(0, width_padsize)), 'reflect') 
    
    dsm = None
    seg_result = None
    seg_result = np.zeros((num_seg_classes, bgr0.shape[0], bgr0.shape[1]),dtype=np.float32)
    for y in xrange(0, bgr0.shape[0], step_size):
        for x in xrange(0, bgr0.shape[1], step_size):
            # yield the current window
            im = bgr0[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE, :]
            if dsm0 is not None:
                dsm = dsm0[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
            # begin detection in a 500x500 image
            seg_ = get_seg_result(net, im, bgr_mean, dsm, dsm_mean, prob_blob_name)
            
            if seg_ is not None:
                seg_result[:, y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = seg_ 
                
    return seg_result[:, 0:height, 0:width]

if __name__ == '__main__':

    args = parse_args() 
    
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    bgr_mean = literal_eval(args.bgr_mean)
    dsm_mean = None
    if args.dsm_mean is not None:
        dsm_mean = literal_eval(args.dsm_mean)
    
    # load net
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    
    data_path = os.path.join(args.devkit_path, 'VOC' + args.year)
    image_set_file = os.path.join(data_path, 'ImageSets', 'Main',
                                  args.image_set + '.txt')
    image_index = []
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    
    num_images = len(image_index)
    all_boxes = [[] for _ in xrange(num_images)]
    
    for i in xrange(num_images):
        index = image_index[i]
        bgr_image_path = os.path.join(data_path, 'JPEGImages', index + '.jpg')
        dsm_image_path = os.path.join(data_path, 'JPEGImages', index + '_depth.jpg')
        bgr0 = cv2.imread(bgr_image_path) #
        dsm0 = None
        if dsm_mean is not None:  
            dsm0 = cv2.imread(dsm_image_path, cv2.IMREAD_GRAYSCALE) 
            
        seg_result = test_on_one_image(net, bgr0, bgr_mean, dsm0, dsm_mean, args.step_size, prob_blob_name=args.prob_blob_name)
        
        if seg_result is not None:
            filename = args.save_prefix + '_det_' + args.image_set + '_' + index + '.mat'
            save_filepath = os.path.join(args.devkit_path,
                                         'results',
                                         'VOC' + args.year,
                                         'Main',
                                         filename) 
            scipy.io.savemat(save_filepath, {'seg_prob': seg_result})
        
    
