% This code is in association with the following paper
% "Ma J, Zhou Z, Wang B, et al. Infrared and visible image fusion based on visual saliency map and weighted least square optimization[J].
% Infrared Physics & Technology, 2017, 82:8-17."
% Authors: Jinlei Ma, Zhiqiang Zhou, Bo Wang, Hua Zong
% Code edited by Jinlei Ma, email: majinlei121@163.com

% clear all
% close all

%Test
% start = 1197;
% ending = 3667;

% for k = start:ending
%     rgb_FileName = sprintf('test/%d_RGB.jpg', k);
%     if isfile(rgb_FileName)
%         json_FileName = sprintf('test/%d_GT.json', k);
%         copyfile(json_FileName,'test_fused\');
%     end
% end

%  for k = start:ending
%     rgb_FileName = sprintf('test/%d_RGB.jpg', k);
%     t_FileName = sprintf('test/%d_T.jpg', k);
%         
%     if isfile(rgb_FileName)&&isfile(t_FileName)
%         fprintf('Processing file %d.\n',k);
%         rgb_img = double(imread(rgb_FileName))/255;
%         t_img = double(imread(t_FileName))/255;
%          
%       else
%  		fprintf('File %d does not exist.\n',k);
%         continue;
%     end
%     fprintf('size(rgb_img) is [%s]\n', int2str(size(rgb_img)));
%         fprintf('size(t_img) is [%s]\n', int2str(size(t_img)));
%     rgb_img = rgb2gray(rgb_img);
%     %rgb_img = imresize(rgb_img,[224, 224]);
%     t_img = rgb2gray(t_img);
%     fused = WLS_Fusion(rgb_img,t_img);
%     fused_path = sprintf('normal_test/%d_F.jpg', k);
%     imwrite(fused,fused_path);
% end

%Train
% start = 1162;
% ending = 3815;

% for k = start:ending
%      rgb_FileName = sprintf('train/%d_RGB.jpg', k);
%      if isfile(rgb_FileName)
%          json_FileName = sprintf('train/%d_GT.json', k);
%          copyfile(json_FileName,'train_fused\');
%      end
% end

% img_count = 0;
% for k = start:ending
%     rgb_FileName = sprintf('train/%d_RGB.jpg', k);
%     t_FileName = sprintf('train/%d_T.jpg', k);
%        
%     if isfile(rgb_FileName)&&isfile(t_FileName)
%         img_count = img_count + 1;
%         fprintf('Processing file %d.\n',k);
%         rgb_img = double(imread(rgb_FileName))/255;
%         t_img = double(imread(t_FileName))/255;
%         fprintf('size(rgb_img) is [%s]\n', int2str(size(rgb_img)));
%         fprintf('size(t_img) is [%s]\n', int2str(size(t_img)));
%         if isequal(size(rgb_img),size(t_img))==false
%             %fprintf('Here\n')
%             t_img = imrotate(t_img,90);
%         end
%     else
%         
% 		fprintf('File %d does not exist.\n',k);
%         continue;
%     
%     end
%     fprintf('size(rgb_img) is [%s]\n', int2str(size(rgb_img)));
%         fprintf('size(t_img) is [%s]\n', int2str(size(t_img)));
%     rgb_img = rgb2gray(rgb_img);
%     t_img = rgb2gray(t_img);
%     fused = WLS_Fusion(rgb_img,t_img);
%     fused_path = sprintf('normal_train/%d_F.jpg', k);
%     imwrite(fused,fused_path);
%     
% end
% 
% fprintf('img_count is [%d]\n', img_count);


%val
start = 1157;
ending = 3816;


img_count = 0;
for k = start:ending
    rgb_FileName = sprintf('val/%d_RGB.jpg', k);
    t_FileName = sprintf('val/%d_T.jpg', k);
       
    if isfile(rgb_FileName)&&isfile(t_FileName)
        img_count = img_count + 1;
        fprintf('Processing file %d.\n',k);
        rgb_img = double(imread(rgb_FileName))/255;
        t_img = double(imread(t_FileName))/255;
%         fprintf('size(rgb_img) is [%s]\n', int2str(size(rgb_img)));
%         fprintf('size(t_img) is [%s]\n', int2str(size(t_img)));
        if isequal(size(rgb_img),size(t_img))==false
            %fprintf('Here\n')
            t_img = imrotate(t_img,90);
        end
     else
		fprintf('File %d does not exist.\n',k);
        continue;
    end
    fprintf('size(rgb_img) is [%s]\n', int2str(size(rgb_img)));
        fprintf('size(t_img) is [%s]\n', int2str(size(t_img)));
    rgb_img = rgb2gray(rgb_img);
    t_img = rgb2gray(t_img);
    fused = WLS_Fusion(rgb_img,t_img);
    fused_path = sprintf('normal_val/%d_F.jpg', k);
    imwrite(fused,fused_path);
end

fprintf('img_count is [%d]\n', img_count);     
    
    




