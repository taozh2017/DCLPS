import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and VGG")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")
        # parser.add_argument("--target", type=str, default='/opt/data/private/datasets/Polyp/ETIS_split', help="target dataset")
        parser.add_argument("--target", type=str, default='', help="target dataset")
        parser.add_argument("--data-size", type=int, default=256, help="training data size")
        parser.add_argument("--pseudo-dir", type=str, default='', help="target dataset")

        parser.add_argument("--num-classes", type=int, default=1, help="Number of classes for cityscapes.")
        parser.add_argument("--set", type=str, default='val', help="choose test set.")
        parser.add_argument("--restore-opt1", type=str, default='', help="restore model parameters from beta1")
        parser.add_argument("--prototypes-dir", type=str, default='', help="")
        parser.add_argument("--thres", type=float, default=0.5, help="threshold for pseudo labels.")

        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="restore model parameters from")      

        return parser.parse_args()

