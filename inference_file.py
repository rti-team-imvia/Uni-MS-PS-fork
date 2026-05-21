import os
import argparse
from utils import load_model
from run import run


parser = argparse.ArgumentParser()
parser.add_argument("--nb_img", type=int, default=-1, help="number of images")
parser.add_argument("--folder_save", type=str,
                    default='inference', help="path_to_save_results")
parser.add_argument("--path_obj", type=str,
                    required=True, help= 'path_to_imgs')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--calibrated', action='store_true')
parser.add_argument('--batch_encoder', type=int, default=3, help='batch size for encoder (default 3, recommended 9 for TENDUR)')
parser.add_argument('--batch_transformer', type=int, default=5000, help='batch size for transformer (default 5000, recommended 10000-20000)')
args = parser.parse_args()
        

mode_inference = True

# Normalize path and extract object name
normalized_path = os.path.normpath(args.path_obj)
obj_name = os.path.basename(normalized_path)
if len(obj_name)==0:
    obj_name = os.path.basename(os.path.dirname(normalized_path))
    
model = load_model(path_weight="weights",
                   cuda=args.cuda,
                   mode_inference=mode_inference,
                   calibrated=args.calibrated,
                   batch_size_encoder=args.batch_encoder,
                   batch_size_transformer=args.batch_transformer)

run(model=model,
    path_obj=args.path_obj,
    nb_img=args.nb_img,
    folder_save=args.folder_save,
    obj_name=obj_name,
    calibrated=args.calibrated)