function gen_test_net_scripts

islocalserver = 0;

if 0
    DATASET='Vaihingen';
    if islocalserver
        ROOT='/media/slave1data/rs/isprs2013/Vaihingen/ISPRS_semantic_labeling_Vaihingen';
    else
        ROOT = '/home/nlpr2/data/isprs2013/Vaihingen/';
    end
    RGB_DIR=[ROOT, '/top'];
    GTS_DIR=[ROOT, '/gts_for_participants'];
    DSM_DIR=[ROOT, '/nDSM'];
    PREFIX='top_mosaic_09cm_area';
    trnlist={'1' '3' '5' '7' '13' '17' '21' '23' '26' '32' '37'};
    vallist={'11' '15' '28' '30' '34'};
    tstlist={'2' '4' '6' '8' '10' '12' '14' '16' '20' '22' '24' '27' '29' '31' '33' '35' '38'};
    year = 2014;
else
    DATASET='Potsdam';
    if islocalserver
        ROOT='/media/slave1data/rs/isprs2013/Potsdam/';
    else
        ROOT = '/home/nlpr2/data/isprs2013/Potsdam/';
    end
    RGB_DIR=[ROOT, '/2_Ortho_RGB'];
    GTS_DIR=[ROOT, '/5_Labels_for_participants'];
    DSM_DIR=[ROOT, '/1_DSM_normalisation'];
    PREFIX='top_potsdam_';
    trnlist = {'2_10' '2_12' '3_10' '3_12' '4_10' '4_12' '5_10' '5_11' '6_7' '6_8' '6_10' '6_11' '6_12' '7_7' '7_8' '7_10' '7_11' '7_12'};
    vallist = {'2_11' '3_11' '4_11' '5_11' '6_9' '7_9'};
    tstlist = {'2_13' '2_14' '3_13' '3_14' '4_13' '4_14' '4_15' '5_13' '5_14' '5_15' '6_13' '6_14' '6_15' '7_13'};
    year = 2013;
end

if islocalserver
    %% for RGB ZF
    prototxt = 'models/potsdam/ZF/faster_rcnn_end2end/test.prototxt';
    caffemodel = 'output/faster_rcnn_end2end_update_means/voc_2014_train/zf_faster_rcnn_iter_70000.caffemodel';
    f = fopen(sprintf('test_net_on_%s_RGB_ZF.sh', DATASET), 'w');
    for i = 1:length(vallist)
        prefix = vallist{i};
        fprintf(f, 'python tools/demo_rgbd.py --gpu 0 --prototxt %s --caffemodel %s --rgb_filepath %s/%s%s.png --dsm_filepath %s/%s%s.png --save_prefix RGB_ZF --set PIXEL_MEANS ''[[[84,86,117]]]''\n',...
            prototxt, caffemodel, RGB_DIR, PREFIX, prefix, DSM_DIR, PREFIX, prefix);
    end
    fclose(f);
    
    %% for RGBD ZF
    prototxt = 'models/potsdam_d/ZF/faster_rcnn_end2end/test.prototxt';
    caffemodel = 'output/faster_rcnn_end2end_update_means/voc_2014_train/d_zf_faster_rcnn_iter_70000.caffemodel';
    f = fopen(sprintf('test_net_on_%s_RGBD_ZF.sh', DATASET), 'w');
    for i = 1:length(vallist)
        prefix = vallist{i};
        fprintf(f, 'python tools/demo_rgbd.py --gpu 0 --prototxt %s --caffemodel %s --rgb_filepath %s/%s%s.png --dsm_filepath %s/%s%s.png --save_prefix RGBD_ZF --set TEST.HAS_DSM True PIXEL_MEANS ''[[[84,86,117]]]'' DSM_MEANS ''[32]''\n',...
            prototxt, caffemodel, RGB_DIR, PREFIX, prefix, DSM_DIR, PREFIX, prefix);
    end
    fclose(f);
    
else
    
    if 0
        %% for RGBD ZF2
        prototxt = 'models/potsdam_d/ZF2/faster_rcnn_end2end/test.prototxt';
        caffemodel = 'output/faster_rcnn_end2end_update_means/voc_2014_train/d_zf2_faster_rcnn_iter_70000.caffemodel';
        f = fopen(sprintf('test_net_on_%s_RGBD_ZF2.sh', DATASET), 'w');
        for i = 1:length(vallist)
            prefix = vallist{i};
            fprintf(f, 'python tools/demo_rgbd.py --gpu 1 --prototxt %s --caffemodel %s --rgb_filepath %s/%s%s.png --dsm_filepath %s/%s%s.png --save_prefix RGBD_ZF2 --set TEST.HAS_DSM True PIXEL_MEANS ''[[[84,86,117]]]'' DSM_MEANS ''[32]''\n',...
                prototxt, caffemodel, RGB_DIR, PREFIX, prefix, DSM_DIR, PREFIX, prefix);
        end
        fclose(f);
    end
    
    if 1
        %% for RGB VGG16
        prototxt = 'models/potsdam/VGG16/faster_rcnn_end2end/test.prototxt';
        caffemodel = 'output/faster_rcnn_end2end/voc_2014_train/vgg16_faster_rcnn_iter_70000.caffemodel';
        f = fopen(sprintf('test_net_on_%s_RGB_VGG16.sh', DATASET), 'w');
        for i = 1:length(vallist)
            prefix = vallist{i};
            fprintf(f, 'python tools/demo_rgbd.py --gpu 2 --prototxt %s --caffemodel %s --rgb_filepath %s/%s%s.png --dsm_filepath %s/%s%s.png --save_prefix RGB_VGG16 --nms_threshold 0.7 --set PIXEL_MEANS ''[[[88,94,87]]]''\n',...
                prototxt, caffemodel, RGB_DIR, PREFIX, prefix, DSM_DIR, PREFIX, prefix);
        end
        fclose(f);
    end
    
    if 0
        %% for RGBD VGG16
        prototxt = 'models/potsdam_d/VGG161/faster_rcnn_end2end/test.prototxt';
        caffemodel = 'output/faster_rcnn_end2end_update_means/voc_2014_train/d_vgg161_faster_rcnn_iter_70000.caffemodel';
        f = fopen(sprintf('test_net_on_%s_RGBD_VGG161.sh', DATASET), 'w');
        for i = 1:length(vallist)
            prefix = vallist{i};
            fprintf(f, 'python tools/demo_rgbd.py --gpu 3 --prototxt %s --caffemodel %s --rgb_filepath %s/%s%s.png --dsm_filepath %s/%s%s.png --save_prefix RGBD_VGG161 --set TEST.HAS_DSM True PIXEL_MEANS ''[[[84,86,117]]]'' DSM_MEANS ''[32]''\n',...
                prototxt, caffemodel, RGB_DIR, PREFIX, prefix, DSM_DIR, PREFIX, prefix);
        end
        fclose(f);
    end
end

end

function write_to_file()

end





