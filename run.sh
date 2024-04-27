# E2C  
python -u train_source.py --source '/opt/data/private/datasets/Polyp/ETIS' --target '/opt/data/private/datasets/Polyp/EndoScene' --snapshot-dir './checkpoints_E2C/source' --LB 0.01 --num-steps-stop 10000 --num-steps 10000 --print-freq 50 --learning-rate 2.5e-4 
wait
python -u train_contrast.py --source '/opt/data/private/datasets/Polyp/ETIS' --target '/opt/data/private/datasets/Polyp/EndoScene' --snapshot-dir './checkpoints_E2C/contrast --restore-from './checkpoints_E2C/source/BEST_TRAIN.pth' --output-level --aux --LB 0.01 --num-steps-stop 5000 --num-steps 5000 --print-freq 50 --learning-rate 1e-4 
wait
python -u getSudoLabel.py --target '/opt/data/private/datasets/Polyp/EndoScene' --restore-opt1 './checkpoints_E2C/contrast/BEST_TRAIN.pth' --pseudo-dir '/E2C/' 
wait
python -u self-training.py --target '/opt/data/private/datasets/Polyp/EndoScene' --snapshot-dir './checkpoints_E2C/target' --restore-from './checkpoints_E2C/contrast/BEST_TRAIN.pth' --pseudo-dir '/E2C/' --prototype_dir './checkpoints_E2C/contrast/BEST_TRAIN_PROTOTYPE.pth' --pro-plan 'C' --learning-rate 1e-4 --loss wbce --mpcl-weight 0.0 --num-steps 1000 --num-steps-stop 1000 --print-freq 50 --save-pred-every 1000 


