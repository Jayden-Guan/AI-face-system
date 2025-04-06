import imgaug.augmenters as iaa
import cv2
import os

# define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip horizontally
    #iaa.Crop(percent=(0, 0.1)),  # crop images
    iaa.GaussianBlur(sigma=(0, 3.0)),  # apply gaussian blur
    iaa.Affine(rotate=(-10, 10)),  # rotate images
    iaa.Multiply((0.5, 1.5), per_channel=0.5), # multiply image with random values
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.GammaContrast(gamma=(0.5, 1.5)),
    #iaa.ElasticTransformation(alpha=(0, 50), sigma=(4, 6)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.WithChannels(0, iaa.Add((-50, 50))),
    iaa.WithChannels(1, iaa.Add((-50, 50))),
    iaa.WithChannels(2, iaa.Add((-50, 50))),
    #iaa.CropAndPad(percent=(-0.2, 0.2)),
])

input_dir = "D:/Validation"
output_dir = "C:/Users/Aman/PycharmProjects/HongwenAIProjectFaceAttendance/Valid"

# loop through the folders containing images
for foldername in os.listdir(input_dir):
    # join the path for each folder
    folder_path = os.path.join(input_dir, foldername)

    # create a directory for augmented images with the same name as the folder
    output_folder_path = os.path.join(output_dir, foldername)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    # loop through the images in the folder
    for filename in os.listdir(folder_path):
        # read the image

        image_path = os.path.join(folder_path, filename)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        # apply the augmentation sequence to generate 10 augmented images per original image
        for i in range(10):
            aug_img = seq(image=image)
            # save the augmented image with the same name as the original image
            output_filename = os.path.splitext(filename)[0] + f"_aug_{i}" + os.path.splitext(filename)[1]
            output_path = os.path.join(output_folder_path, output_filename)
            cv2.imwrite(output_path, aug_img)
