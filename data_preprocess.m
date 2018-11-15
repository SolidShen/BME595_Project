% Preprocess the training data and the validation data
% 2d: means each slice is averaged from 0 to 1
% 3d: means each volume is average from 0 to 1
%% 0 Default Setting
clear all;
addpath /home/libilab/a/shared/code/eegfmritool;
load_data_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/images';
load_label_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/labels';
flag_2d = 1;
if flag_2d == 1
    save_data_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/sorted_images_2d';
    save_label_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/sorted_labels_2d';
else
    save_data_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/sorted_images_3d';
    save_label_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/sorted_labels_3d';
end
num_subject = 30;
%% 1 Convert all nifti files into csv files with proper name
for i = 1 : num_subject 
    if flag_2d == 1
        %-- for images
        %-- load the nifti file
        niifile = amri_file_loadnii([load_data_path filesep 'sub' num2str(i) '_img.nii.gz']);
        %-- 4D images
        img = niifile.img;
        %-- convert to double, then to uint8
        img = mat2gray(img);
        for ii = 1 : size(img,3)
            for jj = 1 : size(img,4)
                sorted_img = img(:,:,ii,jj);
                % normalize the current slice if 2d
                sorted_img = (sorted_img - min(sorted_img(:)))/(max(sorted_img(:))-min(sorted_img(:)));
                % name: num of sub + num of slices (3rd dimension) + num of time frame (4th dimension)  
                if i < 10
                    n_i = ['0' num2str(i)];
                else
                    n_i = num2str(i);
                end
                if ii < 10
                    n_ii = ['0' num2str(ii)];
                else
                    n_ii = num2str(ii);
                end
                if jj < 10
                    n_jj = ['0' num2str(jj)];
                else
                    n_jj = num2str(jj);
                end
                csvwrite([save_data_path filesep n_i n_ii n_jj '_img.csv'],sorted_img);
            end
        end
        %-- for labels
        %-- load the nifti file
        niifile = amri_file_loadnii([load_label_path filesep 'sub' num2str(i) '_label.nii.gz']);
        %-- 4D images
        img = niifile.img;
        %-- convert to double, then to uint8
        img = mat2gray(img);
        for ii = 1 : size(img,3)
            for jj = 1 : size(img,4)
                sorted_img = img(:,:,ii,jj);
                % name: num of sub + num of slices (3rd dimension) + num of time frame (4th dimension)  
                if i < 10
                    n_i = ['0' num2str(i)];
                else
                    n_i = num2str(i);
                end
                if ii < 10
                    n_ii = ['0' num2str(ii)];
                else
                    n_ii = num2str(ii);
                end
                if jj < 10
                    n_jj = ['0' num2str(jj)];
                else
                    n_jj = num2str(jj);
                end
                csvwrite([save_label_path filesep n_i n_ii n_jj '_label.csv'],sorted_img);
            end
        end
    else
        %-- for images
        %-- load the nifti file
        niifile = amri_file_loadnii([load_data_path filesep 'sub' num2str(i) '_img.nii.gz']);
        %-- 4D images
        img = niifile.img;
        %-- convert to double, then to uint8
        img = mat2gray(img);
        for jj = 1 : size(img,4)
            % normalize the current volume if 3d
            temp = img(:,:,:,jj);
            temp = (temp - min(temp(:)))/(max(temp(:))-min(temp(:)));
            for ii = 1 : size(img,3)
                sorted_img = temp(:,:,ii);
                % name: num of sub + num of slices (3rd dimension) + num of time frame (4th dimension)  
                if i < 10
                    n_i = ['0' num2str(i)];
                else
                    n_i = num2str(i);
                end
                if ii < 10
                    n_ii = ['0' num2str(ii)];
                else
                    n_ii = num2str(ii);
                end
                if jj < 10
                    n_jj = ['0' num2str(jj)];
                else
                    n_jj = num2str(jj);
                end
                csvwrite([save_data_path filesep n_i n_ii n_jj '_img.csv'],sorted_img);
            end
        end
        %-- for labels
        %-- load the nifti file
        niifile = amri_file_loadnii([load_label_path filesep 'sub' num2str(i) '_label.nii.gz']);
        %-- 4D images
        img = niifile.img;
        %-- convert to double, then to uint8
        img = mat2gray(img);
        for jj = 1 : size(img,4)
            for ii = 1 : size(img,3)
                sorted_img = img(:,:,ii,jj);
                % name: num of sub + num of slices (3rd dimension) + num of time frame (4th dimension)  
                if i < 10
                    n_i = ['0' num2str(i)];
                else
                    n_i = num2str(i);
                end
                if ii < 10
                    n_ii = ['0' num2str(ii)];
                else
                    n_ii = num2str(ii);
                end
                if jj < 10
                    n_jj = ['0' num2str(jj)];
                else
                    n_jj = num2str(jj);
                end
                csvwrite([save_label_path filesep n_i n_ii n_jj '_label.csv'],sorted_img);
            end
        end
    end
end
