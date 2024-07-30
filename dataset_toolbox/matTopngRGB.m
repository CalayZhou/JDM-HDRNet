% mat_path = '/home/calay/DATASET/20230910_Scenary/MatFiles/'
% rgb_path = '/home/calay/DATASET/20230910_Scenary/MatFiles_rgb_png/'

mat_path ='/home/calay/DATASET/mat_final/'
rgb_path = '/home/calay/DATASET/rgb/'


if exist(rgb_path)==0 
    mkdir(rgb_path);  
else
    disp('dir is exist');
end
fileExt = '*.mat';
files = dir(fullfile(mat_path,fileExt));
len = size(files,1);
load curvesRGB.mat;
load wavelength.mat;
for i = 1:len
    mat_name = files(i).name;
    [~,name,fileExt] = fileparts(mat_name)
    load(strcat(mat_path, mat_name));
    wl = wavelength(4:95); % dualix
%     wl = wavelengths(4:95); % pmvis
    shading_down =double(data(:,:, 4:95));
    %     shading_down = uint8(shading_down);
    shading_down=shading_down./max(shading_down(:));
    [phg, pwg, chn]=size(shading_down);
    
    % wl=[452.468750000000,456.476806640625,460.611938476563,464.882171630859,469.296295166016,473.863983154297,478.595794677734,483.503540039063,488.600158691406,493.900115966797,499.419464111328,505.176269531250,511.190734863281,517.485717773438,524.086975097656,531.024047851563,538.330505371094,546.045043945313,554.212463378906,562.884643554688,572.122558593750,581.997863769531,592.595520019531,604.017150878906,616.384826660156,629.846679687500,644.583251953125,660.815979003906];
    
    specNum=size(wl,1);
    x=400:1/3:710-1/3; % spline interpolation
    curR=spline(x,responseR,wl);curG=spline(x,responseG,wl);
    curB=spline(x,responseB,wl);curM=ones(specNum,1);
    
    for i=1:specNum-1
        curR2(i)=curR(i)/(wl(i+1)-wl(i));
        curG2(i)=curG(i)/(wl(i+1)-wl(i));
        curB2(i)=curB(i)/(wl(i+1)-wl(i));
    end
    curR2(specNum)=curR(specNum)/(wl(specNum)-wl(specNum-1));
    curG2(specNum)=curG(specNum)/(wl(specNum)-wl(specNum-1));
    curB2(specNum)=curB(specNum)/(wl(specNum)-wl(specNum-1));
    curR2 = curR2.*5;
    curG2 = curG2.*5;
    curB2 = curB2.*5;
    
    
    curRR= repmat(curR2,[phg*pwg 1]);
    curGG= repmat(curG2,[phg*pwg 1]);
    curBB= repmat(curB2,[phg*pwg 1]);
    
    curRR=reshape(curRR,[phg pwg chn]);
    curGG=reshape(curGG,[phg pwg chn]);
    curBB=reshape(curBB,[phg pwg chn]);
    
    % curRR = permute(curRR,[2 1 3]);
    % curGG = permute(curGG,[2 1 3]);
    % curBB = permute(curBB,[2 1 3]);
    
    
    dying_rgb=zeros(phg,pwg,3);
    
    dying_rgb(:,:,1) = sum(shading_down.*curRR, 3); 
    dying_rgb(:,:,2) = sum(shading_down.*curGG, 3);
    dying_rgb(:,:,3) = sum(shading_down.*curBB, 3)*1.5;% 1.5for dualix
    
    dying_rgb = dying_rgb./max(dying_rgb(:));

    dying_rgb = imrotate(dying_rgb, -90);% for dualix
%     dying_rgb = flipdim(dying_rgb,2); % mirror lr， for dualix
    imwrite(dying_rgb, strcat(rgb_path, name, '.png'));
   
end