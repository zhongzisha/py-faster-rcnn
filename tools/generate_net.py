


import sys
import os
import numpy as np
sys.path.insert(0, '/media/slave1temp/detection/py-faster-rcnn/caffe-fast-rcnn/python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net('VGG16-deploy.prototxt', 'VGG16.v2.caffemodel', caffe.TEST)
net1 = caffe.Net('VGG161-deploy.prototxt', caffe.TRAIN)

net1.params['conv1_1'][0].data[:,0:3,:,:] = net.params['conv1_1'][0].data
net1.params['conv1_1'][0].data[:,3,:,:] = np.random.normal(0,0.01,size=(3,3))
net1.params['conv1_1'][1].data[...] = net.params['conv1_1'][1].data[...]

for key in net.params.keys():
    if key != 'conv1_1':
        print key
        for i in xrange(len(net.params[key])):
            net1.params[key][i].data[...] = net.params[key][i].data[...]
            print 'copied {}[{}]'.format(key, i)

net1.save('VGG161.v2.caffemodel')









