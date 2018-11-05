% Preprocess the training data and the validation data
%% 0 Default Setting
clear all;
load_data_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/images';
load_label_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/labels';
save_data_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/sorted_images';
save_label_path = '/home/libilab2/a/users/xwang/Project_DL_Seg/sorted_labels';
num_subject = 30;

%% 1 Convert all nifti files into csv files with proper name
for i = 1 : num_subject 
    %-- for images
    %-- load the nifti file
    niifile = amri_file_loadnii([load_data_path filesep 'sub' num2str(i) '_img.nii.gz']);
    %-- 4D images
    img = niifile.img;
    %-- convert to double, then to uint8
    img = uint8(mat2gray(img)*255);
    for ii = 1 : size(img,3)
        for jj = 1 : size(img,4)
            sorted_img = img(:,:,ii,jj);
            % name: num of sub + num of time frame (4th dimension) + num of slice (3rd dimension) 
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
            csvwrite([save_data_path filesep n_i n_jj n_ii '_img.csv'],sorted_img);
        end
    end
    %-- for labels
    %-- load the nifti file
    niifile = amri_file_loadnii([load_label_path filesep 'sub' num2str(i) '_label.nii.gz']);
    %-- 4D images
    img = niifile.img;
    %-- convert to double, then to uint8
    img = uint8(mat2gray(img)*255);
    for ii = 1 : size(img,3)
        for jj = 1 : size(img,4)
            sorted_img = img(:,:,ii,jj);
            % name: num of sub + num of time frame (4th dimension) + num of slice (3rd dimension) 
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
            csvwrite([save_label_path filesep n_i n_jj n_ii '_label.csv'],sorted_img);
        end
    end
end
