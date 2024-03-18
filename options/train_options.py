import argparse
import os.path as osp

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser( description="training script for FDA" )
        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and DeepLab_Multi")
        parser.add_argument("--LB", type=float, default=0.01, help="beta for FDA")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")
        parser.add_argument("--entW", type=float, default=0.005, help="weight for entropy")
        parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
        parser.add_argument("--switch2entropy", type=int, default=0, help="switch to entropy after this many steps")
        parser.add_argument("--loss", type=str, default='wbce+wiou', help="bce, dice, iou or their combinations")
        parser.add_argument("--mpcl-weight", type=float, default=1.0, help="lambda for mpcl loss")
        parser.add_argument("--snapshot-dir", type=str, default='', help="Where to save snapshots of the model.")
        parser.add_argument("--data-size", type=int, default=256, help="training data size")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
        parser.add_argument("--use_source", action="store_true", help="Whether to use source data during training.")
        parser.add_argument("--train_bn", help="train_bn")
        
        parser.add_argument("--source", type=str, default='', help="")
        parser.add_argument("--target", type=str, default='', help="")

        parser.add_argument("--batch-size", type=int, default=4, help="input batch size.")
        parser.add_argument("--num-steps", type=int, default=30000, help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=30000, help="Number of training steps for early stopping.")
        parser.add_argument("--num-workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")

        parser.add_argument("--num-classes", type=int, default=1, help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str, default='DeepLab_init.pth', help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")

        parser.add_argument("--save-pred-every", type=int, default=1000, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-freq", type=int, default=200, help="print loss and time fequency.")
        # for target 
        parser.add_argument("--prototype_dir", help="dir of prototype")
        parser.add_argument("--pro-plan", help="plans of initialize prototypes")
        parser.add_argument("--pseudo-dir", type=str, help="")
        parser.add_argument("--output-level", action="store_true", help="Whether to use output-level contrastive learning.")
        parser.add_argument("--aux", action="store_true", help="Whether to use output-level cross contrastive learning.")
        parser.add_argument("--aux-mse", action="store_true", help="Whether to use output-level MSE loss.")

        # for proca 
        parser.add_argument("--out-level", action="store_true", help="Whether to use output-level contrastive learning(proca).")
        parser.add_argument("--use-bce", action="store_true", help="Whether to use bce(proca).")

        # for ablation
        parser.add_argument("--seed", type=int, help="random seed.")
        parser.add_argument("--dcl", action="store_true", help="domain-interactive contrasitve learning")
        parser.add_argument("--ccl", action="store_true", help="cross contrastive learning.")

        # ablation dcl
        parser.add_argument("--vcl", action="store_true", help="vanilla contrasitve learning")
        parser.add_argument("--cdcl", action="store_true", help="cross domain contrasitve learning")
        parser.add_argument("--m", type=float, default=0.8, help="prototype update ratio")


        # ablation ccl
        parser.add_argument("--mse", action="store_true", help="mse loss for cross consistency training")
        parser.add_argument("--num", type=int, help="number of aux decoders")

        #  ablation ps

        
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    

