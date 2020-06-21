""" Accuracy measurement. """

class AccuracyAggregator(object):
    def __init__(self):
        self.categories = dict()
        self.images = 0
        self.top1 = 0
        self.top5 = 0
        with open("categories.txt", "r") as fh:
            lines = fh.readlines()
            for line in lines:
                line = line.rstrip()
                jpeg_name, index = line.split()
                self.categories[jpeg_name] = int(index)

    def ground_truth(self, image_path):
        jpeg_name = image_path.split("/")[-1]
        return self.categories[jpeg_name]

    def is_top1(self, tensor, gt):
        return 1 if tensor[0] == gt else 0

    def is_top5(self, tensor, gt):
        return 1 if gt in tensor else 0

    def update(self, image_path, tensor):
        tensor = tensor.argsort()[-5:][::-1]
        gt = self.ground_truth(image_path)
        print(gt, tensor)
        self.top1 += self.is_top1(tensor, gt)
        self.top5 += self.is_top5(tensor, gt)
        self.images += 1

    def report(self):
        top1 = round(self.top1 * 100.0/self.images, 2)
        top5 = round(self.top5 * 100.0/self.images, 2)
        return (top1, top5)
