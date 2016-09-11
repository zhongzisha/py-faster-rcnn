function gen_voc_by_random_step(save_root, year, randomsize, stepsize)

% here, we use random sampling for training 

currentdir=pwd;

if ~exist(save_root,'dir')
    mkdir(save_root);
end
cd(save_root);
mkdir Annotations
mkdir ImageSets
mkdir ImageSets/Main
mkdir ImageSets/Segmentation
mkdir ImageSets/Layout
mkdir JPEGImages
mkdir SegmentationClass
mkdir SegmentationClass_Visualization
mkdir ImageSets_Seg

% cmap = voc_colormap(256);
cmap = [0,0,0;1,1,1;0,0,1;0,1,1;0,1,0;1,1,0;1,0,0];

if 0
    DATASET='Vaihingen';
    ROOT='/media/slave1data/rs/isprs2013/Vaihingen/ISPRS_semantic_labeling_Vaihingen';
    RGB_DIR=[ROOT, '/top'];
    GTS_DIR=[ROOT, '/gts_for_participants'];
    DSM_DIR=[ROOT, '/nDSM'];
    PREFIX='top_mosaic_09cm_area';
    trnlist={'1' '3' '5' '7' '13' '17' '21' '23' '26' '32' '37'};
    vallist={'11' '15' '28' '30' '34'};
    tstlist={'2' '4' '6' '8' '10' '12' '14' '16' '20' '22' '24' '27' '29' '31' '33' '35' '38'};
else
    DATASET='Potsdam';
    ROOT='/media/slave1data/rs/isprs2013/Potsdam/';
    RGB_DIR=[ROOT, '/2_Ortho_RGB'];
    GTS_DIR=[ROOT, '/5_Labels_for_participants'];
    DSM_DIR=[ROOT, '/1_DSM_normalisation'];
    PREFIX='top_potsdam_'; 
    trnlist = {'2_10' '2_12' '3_10' '3_12' '4_10' '4_12' '5_10' '5_11' '6_7' '6_8' '6_10' '6_11' '6_12' '7_7' '7_8' '7_10' '7_11' '7_12'};
    vallist = {'2_11' '3_11' '4_11' '5_11' '6_9' '7_9'};
    tstlist = {'2_13' '2_14' '3_13' '3_14' '4_13' '4_14' '4_15' '5_13' '5_14' '5_15' '6_13' '6_14' '6_15' '7_13'};
end

BLOCK_SIZE = 500;
% num_patches_per_trn_image=1000;
% num_patches_per_val_image=100;
length(trnlist)
length(vallist)
length(tstlist)

idx = 1;
carlabel = 4;
background_label = 5;
num_labels = 6;

% for segmentation task
fp1 = fopen('ImageSets_Seg/train.txt','w');
fp2 = fopen('ImageSets_Seg/train_id.txt','w');
fp3 = fopen('ImageSets_Seg/trainval.txt','w');
fp4 = fopen('ImageSets_Seg/trainval_id.txt','w');
% for detection task
fp5 = fopen('ImageSets/Main/train.txt','w');
fp6 = fopen('ImageSets/Main/trainval.txt','w');
% for segmentation task
fp7 = fopen('ImageSets/Segmentation/train.txt','w');
fp8 = fopen('ImageSets/Segmentation/trainval.txt','w');
rgb_sum = 0;
dsm_sum = 0;
count = 0;

%% for trn set
aa = [];
area_sum = 0;
car_count = 0;
area_mean = 0;

%% the following is to use step sampling
for i=1:length(trnlist)
    RGB = imread(sprintf('%s/%s%s.png', RGB_DIR, PREFIX, trnlist{i}));
    DSM = imread(sprintf('%s/%s%s.png', DSM_DIR, PREFIX, trnlist{i}));
    LAB = imread(sprintf('%s/%s%s.png', GTS_DIR, PREFIX, trnlist{i})) - 1;
    [H,W] = size(LAB);
    h=H-BLOCK_SIZE+1;
    w=W-BLOCK_SIZE+1;
    % [Y,X] = meshgrid(1:50:h, 1:50:w);
    Y = 1:stepsize:h;
    X = 1:stepsize:w;
    for ind_y = 1:length(Y)
        for ind_x = 1:length(X)
            prefix = prefix_str(year, idx);
            I = RGB(Y(ind_y):Y(ind_y)+BLOCK_SIZE-1, X(ind_x):X(ind_x)+BLOCK_SIZE-1, :);
            dsm = DSM(Y(ind_y):Y(ind_y)+BLOCK_SIZE-1, X(ind_x):X(ind_x)+BLOCK_SIZE-1, :);
            gt = LAB(Y(ind_y):Y(ind_y)+BLOCK_SIZE-1, X(ind_x):X(ind_x)+BLOCK_SIZE-1, :);
            [height, width, depth] = size(I);
            
            %gt = gt + 1;
            %gt(gt==background_label) = 0;
            carmask = gt==carlabel;% car label is 5, background is 6
            s = regionprops(carmask,'Centroid','BoundingBox', 'Area');
            
            if isempty(s)
                continue;
            end
            
            validcarcount = 0;
            for bi = 1:length(s)
                xmin = s(bi).BoundingBox(1);
                ymin = s(bi).BoundingBox(2);
                xmax = xmin + s(bi).BoundingBox(3);
                ymax = ymin + s(bi).BoundingBox(4);
                if xmin<1
                    xmin=1;
                end
                if ymin<1
                    ymin=1;
                end
                if xmax>width
                    xmax = width;
                end
                if ymax>height
                    ymax=height;
                end
                xmin = ceil(xmin);
                xmax = ceil(xmax);
                ymin = ceil(ymin);
                ymax = ceil(ymax);
                
                if xmin>1 && ymin>1 && xmax<width && ymax<height
                    validcarcount = validcarcount + 1;
                elseif s(bi).Area > 0.75 * area_mean
                    validcarcount = validcarcount + 1;
                end
                
            end
            
            if validcarcount==0
                continue
            end
            
            % now, the image is a valid image, do the following things
            
            fpa = fopen(sprintf('Annotations/%s.xml', prefix),'w');
            fprintf(fpa,'<annotation>\n');
            fprintf(fpa,'<folder>VOC%d</folder>\n',year);
            fprintf(fpa,'<filename>%s.jpg</filename>\n',prefix);
            fprintf(fpa,'<source>\n');
            fprintf(fpa,'<database>The VOC%d Database</database>\n',year);
            fprintf(fpa,'<annotation>PASCAL VOC%d</annotation>\n',year);
            fprintf(fpa,'<image>none</image>\n');
            fprintf(fpa,'<flickrid>0</flickrid>\n');
            fprintf(fpa,'</source>\n');
            fprintf(fpa,'<owner>\n');
            fprintf(fpa,'<flickrid>isprs</flickrid>\n');
            fprintf(fpa,'<name>isprs</name>\n');
            fprintf(fpa,'</owner>\n');
            fprintf(fpa,'<size><width>%d</width><height>%d</height><depth>%d</depth></size>\n',width,height,depth);
            fprintf(fpa,'<segmented>0</segmented>\n');
            for bi = 1:length(s)
                xmin = s(bi).BoundingBox(1);
                ymin = s(bi).BoundingBox(2);
                xmax = xmin + s(bi).BoundingBox(3);
                ymax = ymin + s(bi).BoundingBox(4);
                if xmin<1
                    xmin=1;
                end
                if ymin<1
                    ymin=1;
                end
                if xmax>width
                    xmax = width;
                end
                if ymax>height
                    ymax=height;
                end
                xmin = ceil(xmin);
                xmax = ceil(xmax);
                ymin = ceil(ymin);
                ymax = ceil(ymax);
                
                if xmin>1 && ymin>1 && xmax<width && ymax<height
                    fprintf(fpa,'<object>\n');
                    fprintf(fpa,'<name>car</name>\n');
                    fprintf(fpa,'<pose>Left</pose>\n');
                    fprintf(fpa,'<truncated>0</truncated>\n');
                    fprintf(fpa,'<difficult>0</difficult>\n');
                    fprintf(fpa,'<bndbox>\n');
                    fprintf(fpa, '<xmin>%d</xmin>\n', xmin);
                    fprintf(fpa, '<ymin>%d</ymin>\n', ymin);
                    fprintf(fpa, '<xmax>%d</xmax>\n', xmax);
                    fprintf(fpa, '<ymax>%d</ymax>\n', ymax);
                    fprintf(fpa,'</bndbox>\n');
                    fprintf(fpa,'</object>\n');
                    
                    car_count = car_count + 1;
                    area_sum = area_sum + s(bi).Area;
                    area_mean  = area_sum / car_count;
                elseif s(bi).Area > 0.75 * area_mean
                    fprintf(fpa,'<object>\n');
                    fprintf(fpa,'<name>car</name>\n');
                    fprintf(fpa,'<pose>Left</pose>\n');
                    fprintf(fpa,'<truncated>0</truncated>\n');
                    fprintf(fpa,'<difficult>0</difficult>\n');
                    fprintf(fpa,'<bndbox>\n');
                    fprintf(fpa, '<xmin>%d</xmin>\n', xmin);
                    fprintf(fpa, '<ymin>%d</ymin>\n', ymin);
                    fprintf(fpa, '<xmax>%d</xmax>\n', xmax);
                    fprintf(fpa, '<ymax>%d</ymax>\n', ymax);
                    fprintf(fpa,'</bndbox>\n');
                    fprintf(fpa,'</object>\n');
                    
                    car_count = car_count + 1;
                    area_sum = area_sum + s(bi).Area;
                    area_mean  = area_sum / car_count;
                end
            end
            fprintf(fpa,'</annotation>\n');
            fclose(fpa);
            fprintf(fp5, '%s\n', prefix);
            fprintf(fp6, '%s\n', prefix);
            
            rgb_sum = rgb_sum + sum(reshape(double(I), size(I,1)*size(I,2), 3));
            dsm_sum = dsm_sum + sum(double(dsm(:)));
            count = count + size(I,1)*size(I,2);
            imwrite(I, sprintf('JPEGImages/%s.jpg', prefix));
            imwrite(dsm, sprintf('JPEGImages/%s_depth.jpg', prefix));
            imwrite(uint8(gt+1),cmap,sprintf('SegmentationClass_Visualization/%s.png', prefix));
            imwrite(uint8(gt),sprintf('SegmentationClass/%s.png', prefix));
            fprintf(fp7, '%s\n', prefix);
            fprintf(fp8, '%s\n', prefix);
            
            % for computing weights
            [a,~] = imhist(gt);
            aa = cat(2, aa, a(1:num_labels));
            
            fprintf(fp1, '/JPEGImages/%s.jpg /SegmentationClass/%s.png\n', prefix, prefix);
            fprintf(fp2, '%s\n', prefix);
            fprintf(fp3, '/JPEGImages/%s.jpg /SegmentationClass/%s.png\n', prefix, prefix);
            fprintf(fp4, '%s\n', prefix);
            
            idx = idx + 1;
        end
    end
end

%% the following is to use random sampling 
for i=1:length(trnlist)
    RGB = imread(sprintf('%s/%s%s.png', RGB_DIR, PREFIX, trnlist{i}));
    DSM = imread(sprintf('%s/%s%s.png', DSM_DIR, PREFIX, trnlist{i}));
    LAB = imread(sprintf('%s/%s%s.png', GTS_DIR, PREFIX, trnlist{i})) - 1;
    [H,W] = size(LAB);
    h=H-BLOCK_SIZE+1;
    w=W-BLOCK_SIZE+1;
    [Y,X] = meshgrid(1:h, 1:w);
    indexes = randperm(h*w);
    for j=1:randomsize
        ind = indexes(j);
        prefix = prefix_str(year, idx);
        I = RGB(Y(ind):Y(ind)+BLOCK_SIZE-1, X(ind):X(ind)+BLOCK_SIZE-1, :);
        dsm = DSM(Y(ind):Y(ind)+BLOCK_SIZE-1, X(ind):X(ind)+BLOCK_SIZE-1, :);
        gt = LAB(Y(ind):Y(ind)+BLOCK_SIZE-1, X(ind):X(ind)+BLOCK_SIZE-1, :);
        
        [height, width, depth] = size(I);
        
        %gt = gt + 1;
        %gt(gt==background_label) = 0;
        carmask = gt==carlabel;% car label is 5, background is 6
        s = regionprops(carmask,'Centroid','BoundingBox', 'Area');
        
        if isempty(s)
            continue;
        end
        
        validcarcount = 0;
        for bi = 1:length(s)
            xmin = s(bi).BoundingBox(1);
            ymin = s(bi).BoundingBox(2);
            xmax = xmin + s(bi).BoundingBox(3);
            ymax = ymin + s(bi).BoundingBox(4);
            if xmin<1
                xmin=1;
            end
            if ymin<1
                ymin=1;
            end
            if xmax>width
                xmax = width;
            end
            if ymax>height
                ymax=height;
            end
            xmin = ceil(xmin);
            xmax = ceil(xmax);
            ymin = ceil(ymin);
            ymax = ceil(ymax);  
            
            if xmin>1 && ymin>1 && xmax<width && ymax<height
                validcarcount = validcarcount + 1;
            elseif s(bi).Area > 0.75 * area_mean
                validcarcount = validcarcount + 1;
            end 
                
        end
        
        if validcarcount==0
            continue
        end
        
        % now, the image is a valid image, do the following things
        
        fpa = fopen(sprintf('Annotations/%s.xml', prefix),'w');
        fprintf(fpa,'<annotation>\n');
        fprintf(fpa,'<folder>VOC%d</folder>\n',year);
        fprintf(fpa,'<filename>%s.jpg</filename>\n',prefix);
        fprintf(fpa,'<source>\n');
        fprintf(fpa,'<database>The VOC%d Database</database>\n',year);
        fprintf(fpa,'<annotation>PASCAL VOC%d</annotation>\n',year);
        fprintf(fpa,'<image>none</image>\n');
        fprintf(fpa,'<flickrid>0</flickrid>\n');
        fprintf(fpa,'</source>\n');
        fprintf(fpa,'<owner>\n');
        fprintf(fpa,'<flickrid>isprs</flickrid>\n');
        fprintf(fpa,'<name>isprs</name>\n');
        fprintf(fpa,'</owner>\n');
        fprintf(fpa,'<size><width>%d</width><height>%d</height><depth>%d</depth></size>\n',width,height,depth);
        fprintf(fpa,'<segmented>0</segmented>\n');
        for bi = 1:length(s)
            xmin = s(bi).BoundingBox(1);
            ymin = s(bi).BoundingBox(2);
            xmax = xmin + s(bi).BoundingBox(3);
            ymax = ymin + s(bi).BoundingBox(4);
            if xmin<1
                xmin=1;
            end
            if ymin<1
                ymin=1;
            end
            if xmax>width
                xmax = width;
            end
            if ymax>height
                ymax=height;
            end
            xmin = ceil(xmin);
            xmax = ceil(xmax);
            ymin = ceil(ymin);
            ymax = ceil(ymax);
            
            if xmin>1 && ymin>1 && xmax<width && ymax<height
                fprintf(fpa,'<object>\n');
                fprintf(fpa,'<name>car</name>\n');
                fprintf(fpa,'<pose>Left</pose>\n');
                fprintf(fpa,'<truncated>0</truncated>\n');
                fprintf(fpa,'<difficult>0</difficult>\n');
                fprintf(fpa,'<bndbox>\n');
                fprintf(fpa, '<xmin>%d</xmin>\n', xmin);
                fprintf(fpa, '<ymin>%d</ymin>\n', ymin);
                fprintf(fpa, '<xmax>%d</xmax>\n', xmax);
                fprintf(fpa, '<ymax>%d</ymax>\n', ymax);
                fprintf(fpa,'</bndbox>\n');
                fprintf(fpa,'</object>\n');
                
                car_count = car_count + 1;
                area_sum = area_sum + s(bi).Area;
                area_mean  = area_sum / car_count;
            elseif s(bi).Area > 0.75 * area_mean
                fprintf(fpa,'<object>\n');
                fprintf(fpa,'<name>car</name>\n');
                fprintf(fpa,'<pose>Left</pose>\n');
                fprintf(fpa,'<truncated>0</truncated>\n');
                fprintf(fpa,'<difficult>0</difficult>\n');
                fprintf(fpa,'<bndbox>\n');
                fprintf(fpa, '<xmin>%d</xmin>\n', xmin);
                fprintf(fpa, '<ymin>%d</ymin>\n', ymin);
                fprintf(fpa, '<xmax>%d</xmax>\n', xmax);
                fprintf(fpa, '<ymax>%d</ymax>\n', ymax);
                fprintf(fpa,'</bndbox>\n');
                fprintf(fpa,'</object>\n');
                
                car_count = car_count + 1;
                area_sum = area_sum + s(bi).Area;
                area_mean  = area_sum / car_count;
            end
        end
        fprintf(fpa,'</annotation>\n');
        fclose(fpa);
        fprintf(fp5, '%s\n', prefix);
        fprintf(fp6, '%s\n', prefix); 
        
        rgb_sum = rgb_sum + sum(reshape(double(I), size(I,1)*size(I,2), 3));
        dsm_sum = dsm_sum + sum(double(dsm(:)));
        count = count + size(I,1)*size(I,2);
        imwrite(I, sprintf('JPEGImages/%s.jpg', prefix));
        imwrite(dsm, sprintf('JPEGImages/%s_depth.jpg', prefix));
        imwrite(uint8(gt+1),cmap,sprintf('SegmentationClass_Visualization/%s.png', prefix));
        imwrite(uint8(gt),sprintf('SegmentationClass/%s.png', prefix));
        fprintf(fp7, '%s\n', prefix);
        fprintf(fp8, '%s\n', prefix);
        
        % for computing weights
        [a,~] = imhist(gt);
        aa = cat(2, aa, a(1:num_labels));
        
        fprintf(fp1, '/JPEGImages/%s.jpg /SegmentationClass/%s.png\n', prefix, prefix);
        fprintf(fp2, '%s\n', prefix);
        fprintf(fp3, '/JPEGImages/%s.jpg /SegmentationClass/%s.png\n', prefix, prefix);
        fprintf(fp4, '%s\n', prefix);
        
        idx = idx + 1;
    end
end

fclose(fp1);
fclose(fp2);
fclose(fp5);
fclose(fp7);

rgb_mean1 = rgb_sum / count
dsm_mean1 = dsm_sum / count

cc=zeros(num_labels,1);
for i=1:num_labels
    index = find(aa(i,:)~=0);
    cc(i)=length(index);
end
cc=cc.*(BLOCK_SIZE*BLOCK_SIZE);
bb=sum(aa,2);
dd=bb./cc;
weights = median(dd)./dd
csvwrite(['trn_weights.txt'], weights);

%% for val set
fp1 = fopen('ImageSets_Seg/val.txt','w');
fp2 = fopen('ImageSets_Seg/val_id.txt','w');
fp5 = fopen('ImageSets/Main/val.txt','w');
fp7 = fopen('ImageSets/Segmentation/val.txt','w');

for i=1:length(vallist)
    RGB = imread(sprintf('%s/%s%s.png', RGB_DIR, PREFIX, vallist{i}));
    DSM = imread(sprintf('%s/%s%s.png', DSM_DIR, PREFIX, vallist{i}));
    LAB = imread(sprintf('%s/%s%s.png', GTS_DIR, PREFIX, vallist{i})) - 1;
    [H,W] = size(LAB);
    h=H-BLOCK_SIZE+1;
    w=W-BLOCK_SIZE+1;
    % [Y,X] = meshgrid(1:50:h, 1:50:w);
    Y = 1:stepsize:h;
    X = 1:stepsize:w;
    for ind_y = 1:length(Y)
        for ind_x = 1:length(X)
            prefix = prefix_str(year, idx);
            I = RGB(Y(ind_y):Y(ind_y)+BLOCK_SIZE-1, X(ind_x):X(ind_x)+BLOCK_SIZE-1, :);
            dsm = DSM(Y(ind_y):Y(ind_y)+BLOCK_SIZE-1, X(ind_x):X(ind_x)+BLOCK_SIZE-1, :);
            gt = LAB(Y(ind_y):Y(ind_y)+BLOCK_SIZE-1, X(ind_x):X(ind_x)+BLOCK_SIZE-1, :);
            [height, width, depth] = size(I);
            
            %gt = gt + 1;
            %gt(gt==background_label) = 0;
            carmask = gt==carlabel;% car label is 5, background is 6
            s = regionprops(carmask,'Centroid','BoundingBox', 'Area');
            
            if isempty(s)
                continue;
            end
            
            validcarcount = 0;
            for bi = 1:length(s)
                xmin = s(bi).BoundingBox(1);
                ymin = s(bi).BoundingBox(2);
                xmax = xmin + s(bi).BoundingBox(3);
                ymax = ymin + s(bi).BoundingBox(4);
                if xmin<1
                    xmin=1;
                end
                if ymin<1
                    ymin=1;
                end
                if xmax>width
                    xmax = width;
                end
                if ymax>height
                    ymax=height;
                end
                xmin = ceil(xmin);
                xmax = ceil(xmax);
                ymin = ceil(ymin);
                ymax = ceil(ymax);
                
                if xmin>1 && ymin>1 && xmax<width && ymax<height
                    validcarcount = validcarcount + 1;
                elseif s(bi).Area > 0.75 * area_mean
                    validcarcount = validcarcount + 1;
                end
                
            end
            
            if validcarcount==0
                continue
            end
            
            % now, the image is a valid image, do the following things
            
            fpa = fopen(sprintf('Annotations/%s.xml', prefix),'w');
            fprintf(fpa,'<annotation>\n');
            fprintf(fpa,'<folder>VOC%d</folder>\n',year);
            fprintf(fpa,'<filename>%s.jpg</filename>\n',prefix);
            fprintf(fpa,'<source>\n');
            fprintf(fpa,'<database>The VOC%d Database</database>\n',year);
            fprintf(fpa,'<annotation>PASCAL VOC%d</annotation>\n',year);
            fprintf(fpa,'<image>none</image>\n');
            fprintf(fpa,'<flickrid>0</flickrid>\n');
            fprintf(fpa,'</source>\n');
            fprintf(fpa,'<owner>\n');
            fprintf(fpa,'<flickrid>isprs</flickrid>\n');
            fprintf(fpa,'<name>isprs</name>\n');
            fprintf(fpa,'</owner>\n');
            fprintf(fpa,'<size><width>%d</width><height>%d</height><depth>%d</depth></size>\n',width,height,depth);
            fprintf(fpa,'<segmented>0</segmented>\n');
            for bi = 1:length(s)
                xmin = s(bi).BoundingBox(1);
                ymin = s(bi).BoundingBox(2);
                xmax = xmin + s(bi).BoundingBox(3);
                ymax = ymin + s(bi).BoundingBox(4);
                if xmin<1
                    xmin=1;
                end
                if ymin<1
                    ymin=1;
                end
                if xmax>width
                    xmax = width;
                end
                if ymax>height
                    ymax=height;
                end
                xmin = ceil(xmin);
                xmax = ceil(xmax);
                ymin = ceil(ymin);
                ymax = ceil(ymax);
                
                if xmin>1 && ymin>1 && xmax<width && ymax<height
                    fprintf(fpa,'<object>\n');
                    fprintf(fpa,'<name>car</name>\n');
                    fprintf(fpa,'<pose>Left</pose>\n');
                    fprintf(fpa,'<truncated>0</truncated>\n');
                    fprintf(fpa,'<difficult>0</difficult>\n');
                    fprintf(fpa,'<bndbox>\n');
                    fprintf(fpa, '<xmin>%d</xmin>\n', xmin);
                    fprintf(fpa, '<ymin>%d</ymin>\n', ymin);
                    fprintf(fpa, '<xmax>%d</xmax>\n', xmax);
                    fprintf(fpa, '<ymax>%d</ymax>\n', ymax);
                    fprintf(fpa,'</bndbox>\n');
                    fprintf(fpa,'</object>\n');
                    
                    car_count = car_count + 1;
                    area_sum = area_sum + s(bi).Area;
                    area_mean  = area_sum / car_count;
                elseif s(bi).Area > 0.75 * area_mean
                    fprintf(fpa,'<object>\n');
                    fprintf(fpa,'<name>car</name>\n');
                    fprintf(fpa,'<pose>Left</pose>\n');
                    fprintf(fpa,'<truncated>0</truncated>\n');
                    fprintf(fpa,'<difficult>0</difficult>\n');
                    fprintf(fpa,'<bndbox>\n');
                    fprintf(fpa, '<xmin>%d</xmin>\n', xmin);
                    fprintf(fpa, '<ymin>%d</ymin>\n', ymin);
                    fprintf(fpa, '<xmax>%d</xmax>\n', xmax);
                    fprintf(fpa, '<ymax>%d</ymax>\n', ymax);
                    fprintf(fpa,'</bndbox>\n');
                    fprintf(fpa,'</object>\n');
                    
                    car_count = car_count + 1;
                    area_sum = area_sum + s(bi).Area;
                    area_mean  = area_sum / car_count;
                end
            end
            fprintf(fpa,'</annotation>\n');
            fclose(fpa);
            fprintf(fp5, '%s\n', prefix);
            fprintf(fp6, '%s\n', prefix);
            
            rgb_sum = rgb_sum + sum(reshape(double(I), size(I,1)*size(I,2), 3));
            dsm_sum = dsm_sum + sum(double(dsm(:)));
            count = count + size(I,1)*size(I,2);
            imwrite(I, sprintf('JPEGImages/%s.jpg', prefix));
            imwrite(dsm, sprintf('JPEGImages/%s_depth.jpg', prefix));
            imwrite(uint8(gt+1),cmap,sprintf('SegmentationClass_Visualization/%s.png', prefix));
            imwrite(uint8(gt),sprintf('SegmentationClass/%s.png', prefix));
            fprintf(fp7, '%s\n', prefix);
            fprintf(fp8, '%s\n', prefix);
            
            % for computing weights
            [a,~] = imhist(gt);
            aa = cat(2, aa, a(1:num_labels));
            
            fprintf(fp1, '/JPEGImages/%s.jpg /SegmentationClass/%s.png\n', prefix, prefix);
            fprintf(fp2, '%s\n', prefix);
            fprintf(fp3, '/JPEGImages/%s.jpg /SegmentationClass/%s.png\n', prefix, prefix);
            fprintf(fp4, '%s\n', prefix);
            
            idx = idx + 1;
        end
    end
end
fclose(fp1);
fclose(fp2);
fclose(fp5);
fclose(fp7);

rgb_mean2 = rgb_sum / count
dsm_mean2 = dsm_sum / count

cc=zeros(num_labels,1);
for i=1:num_labels
    index = find(aa(i,:)~=0);
    cc(i)=length(index);
end
cc=cc.*(BLOCK_SIZE*BLOCK_SIZE);
bb=sum(aa,2);
dd=bb./cc;
weights = median(dd)./dd
csvwrite(['trnval_weights.txt'], weights);


fp_mean = fopen('ImageSets_Seg/mean_values.txt','w');
fprintf(fp_mean, 'train(rgb): %.6f, %.6f, %.6f\n', rgb_mean1(1), rgb_mean1(2), rgb_mean1(3));
fprintf(fp_mean, 'train(dsm): %.6f\n', dsm_mean1(1));
fprintf(fp_mean, 'trainval(rgb): %.6f, %.6f, %.6f\n', rgb_mean2(1), rgb_mean2(2), rgb_mean2(3));
fprintf(fp_mean, 'trainval(dsm): %.6f\n', dsm_mean2(1));
fclose(fp_mean);

fclose(fp3);
fclose(fp4);
fclose(fp6);
fclose(fp8);

cd(currentdir);
end

function prefix_str = prefix_str(year, idx)
prefix_str = sprintf('%d_%06d', year, idx);
end

function cmap = voc_colormap(N)

if nargin==0
    N=256
end
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
end





