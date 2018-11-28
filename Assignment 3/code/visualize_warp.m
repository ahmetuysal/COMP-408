img = imread('ahmet.jpg');

[size_x, size_y, ~] = size(img);

tform_reflection = affine2d([ -1 0 0; 0 1 0; 0 0 1]);
tform_shear = affine2d([1 .5 0; 0 1 0; 0 0 1]);
tform_shear2 = affine2d([1 .25 0; .25 1 0; 0 0 1]);
tform_shear3 = affine2d([1 -.5 0; 0 1 0; 0 0 1]);
tform_shear4 = affine2d([1 -.25 0; -.25 1 0; 0 0 1]);
tform_shear_reflect = affine2d([-1 .5 0; 0 1 0; 0 0 1]);
tform_shear_reflect2 = affine2d([-1 -.3 0; .25 1 0; 0 0 1]);

tform = affine2d([ 0.5*cos(pi/4) sin(pi/4)     0;
                  -sin(pi/4)     0.5*cos(pi/4) 0;
                   0             0             1]);
               
transforms = [tform_reflection, tform_shear, tform_shear2, tform_shear3,...
    tform_shear4, tform_shear_reflect, tform_shear_reflect2, tform];

for i = 1:length(transforms)
    transform = transforms(i);
    warper = images.geotrans.Warper(transform, size(img));
    warped_image = warp(warper, img);
    warped_image = imresize(warped_image, [size_x, size_y]);
    imwrite(warped_image, sprintf('ahmet%d.jpg', i))
end