import glob

imagenet_path = '/home/ubuntu/imagenet/val/'
all_class_path = sorted(glob.glob(imagenet_path+'*'))

images = list()
for cur_class in all_class_path:
    all_image = glob.glob(cur_class+'/*')
    images.extend(all_image)

random.seed(0)
random.shuffle(images)


calibration_dataset = images[0:1000]
test_dataset = images[1000:]

