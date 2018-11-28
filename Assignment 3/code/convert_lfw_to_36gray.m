data_path = '../data/'; 
lfw_faces_path = fullfile(data_path, 'lfw_faces');
image_files = dir( fullfile( lfw_faces_path, '*.jpg') );
num_images = length(image_files);

for i = 1:num_images
   img = imread(strcat(image_files(i).folder, '\',...
       image_files(i).name));
   if size(img, 3) == 3
        img = rgb2gray(img);
   end
    
   img = imresize(img, [36, 36]);
   imwrite(img, (fullfile(lfw_faces_path, image_files(i).name)))
   disp((fullfile(lfw_faces_path, image_files(i).name)))
end
