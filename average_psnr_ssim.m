test_path = './outputs/';
gt_path =  '/home/calay/DATASET/Mobile-Spec/eval/target/';

path_list = dir(fullfile(gt_path,'*.tif'));%
img_num = length(path_list);
%calculate psnr
total_psnr = 0;
total_ssim = 0;
total_color = 0;
if img_num > 0 
   for j = 1:img_num 
       image_name = path_list(j).name;
       gt = imread(fullfile(gt_path,image_name));
       input = imread(fullfile(test_path,['epoch1_',num2str(j-1), '.png']));
        psnr_val = psnr(im2double(input), im2double(gt));
       total_psnr = total_psnr + psnr_val;
       
       ssim_val = ssim(input, gt);
       total_ssim = total_ssim + ssim_val;
       
       color = sqrt(sum((rgb2lab(gt) - rgb2lab(input)).^2,3));
       color = mean(color(:));
       total_color = total_color + color;
       fprintf('%d %f %f %f %s\n',j,psnr_val,ssim_val,color,fullfile(test_path,image_name));
   end
end
qm_psnr = total_psnr / img_num;
avg_ssim = total_ssim / img_num;
avg_color = total_color / img_num;
fprintf('The avgerage psnr is: %f', qm_psnr);
fprintf('The avgerage ssim is: %f', avg_ssim);
fprintf('The avgerage lab is: %f', avg_color);
