% This script adapt the CSIQ-VQA database for training NR-SRRED.
clear;
restoredefaultpath;
clc;
addpath("matlabPyrTools/");
% The path of the CSIQ database, please change it to the root of CSIQ database on your computer.
PATH_ABS = "";
PATH_INPUT = "videos/";
% The output path, please also change it to your path.
PATH_OUTPUT = "";
Folders = ["BasketballDrive","BQMall","BQTerrace","Cactus","Carving","Chipmunks","Flowervase","Keiba","Kimono","ParkScene","PartyScene","Timelapse"];

resolution = [480,832];
frameNum = [500,600,600,500,250,240,300,300,240,240,500,300];

idx = 1;
dist_idx_list = [1,2,3,4,5,6,7,8,9,10,11,12,16,17,18];
for ii = 1:length(Folders)
    fn_ref = PATH_ABS+PATH_INPUT+Folders(ii)+"/"+Folders(ii)+"_832x480_ref.yuv";
    for kk = dist_idx_list
        fn_dis = PATH_ABS+PATH_INPUT+Folders(ii)+"/"+Folders(ii)+"_832x480_dst_"+num2str(kk,"%02.f")+".yuv";
        for jj = 1:frameNum(ii)
            tic;
            [ref_Y, ref_U, ref_V] = yuvReadFrame(fn_ref,resolution(2),resolution(1),jj);
            [dis_Y, dis_U, dis_V] = yuvReadFrame(fn_dis,resolution(2),resolution(1),jj);
            spatial_ref = extract_info(ref_Y);
            spatial_dis = extract_info(dis_Y);
            SRRED(idx) = mean2(abs(spatial_ref - spatial_dis));
            split_name = split(fn_dis,".");
            img_name = Folders(ii)+"_832x480_dst_"+num2str(kk,"%02.f")+"_"+num2str(jj)+".png";
            name(idx) = img_name;
            dis_U = imresize(dis_U,2);
            dis_V = imresize(dis_V,2);
            dis(:,:,1) = dis_Y;
            dis(:,:,2) = dis_U;
            dis(:,:,3) = dis_V;
            imwrite(uint8(YUV2RGB(dis)), PATH_OUTPUT+img_name);
            idx = idx + 1;
            toc;
        end
    end
end
save("CSIQ_VQA_name_srred.mat","name","SRRED");
