from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from torchvision.transforms.functional import crop
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from glob import glob

import torch
import argparse
import sys

class inference:
    def __init__(self, max_num_keypoints=2048, depth_confidence=-1, width_confidence=-1):

        file = open("test_data/hero_names.txt", "r").read().split("\n")
        file = [i.split(" ") for i in file]
        self.id_2_hero_names = {k[1]: k[0] for i, k in enumerate(file)}
        self.hero_names_2_id = {k[0]: k[1] for i, k in enumerate(file)}
        hero_icons_file = glob('heroes_test_icon/*.png')
        hero_icon = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device) # load the extractor
        self.matcher = LightGlue(features='superpoint', depth_confidence=depth_confidence, width_confidence=width_confidence).eval().to(self.device) # load the matcher

        for filename in hero_icons_file:
            image = GaussianBlur(5)(load_image(filename))
            feats0 = self.extractor.extract(image.to(self.device))
            hero_name = self.id_2_hero_names[filename.split("/")[1][:-4]]
            hero_icon.append((hero_name, feats0))

        self.hero_icon = hero_icon

    def inference(self, image):
        heros_confidence = dict.fromkeys(self.hero_names_2_id.keys(), 0)
        for icon in self.hero_icon:
            feats1 = self.extractor.extract(image.to(self.device))
            matches01 = self.matcher({"image0": icon[1], "image1": feats1})
            feats0, feats1, matches01 = [
                rbd(x) for x in [icon[1], feats1, matches01]
            ]
            heros_confidence[icon[0]] = matches01["matches"].shape[0]
        heros_confidence = sorted(heros_confidence.items(), key=lambda x: x[1], reverse=True)
        return heros_confidence[0][0]
    
    def evaluate(self, image_path, ground_truth):
        test_data = glob(f'{image_path}/*.jpg')
        test = open(f"{ground_truth}", "r").read().replace("\t", " ").split("\n")
        test = [i.split(" ") for i in test]
        test_data = {i[0]: i[1] for i in test}
        pred = {}

        for filename in tqdm(test_data):
            image = load_image(image_path+filename)
            res = self.inference(image)
            print(f"{filename}: {res}")
            pred[filename] = res[0]

        correct = 0
        with open("result.txt", "w") as f:
            for k, v in pred.items():
                f.write(f"{k} {v}\n")
                if v == test_data[k]:
                    correct += 1
        print(f"Accuracy: {correct/len(pred)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--options", choices=["evaluate", "inference"], required=True)
    parser.add_argument("-t", "--image_test_data", required='evaluate' in sys.argv, help='path to test data')
    parser.add_argument("-g", "--ground_truth", required='evaluate' in sys.argv, help='path to ground truth')

    parser.add_argument("-i", "--image", required='inference' in sys.argv, help="input image")
    parser.add_argument("-m", "--max_num_keypoints", default=2048, type=int, help="max number of keypoints")
    parser.add_argument("-d", "--depth_confidence", default=-1, type=float, help="depth confidence")
    parser.add_argument("-w", "--width_confidence", default=-1, type=float, help="width confidence")
    

    args = parser.parse_args()

    predictor = inference(args.max_num_keypoints, args.depth_confidence, args.width_confidence)

    if args.options == "evaluate":
        predictor.evaluate(args.image_test_data, args.ground_truth)    
    
    if args.options == "inference":
        image = load_image(args.image)
        c, h, w = image.shape
        image = crop(image, 0, 0, h, w//4)

        hero_name = predictor.inference(image)
        print(hero_name)
