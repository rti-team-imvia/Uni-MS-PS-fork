import sys
import torch.nn as nn
import torch
import numpy as np
import time
from tqdm import tqdm
from Transformer_8 import Transformer_8
from utils_process import decrease_size_batch


    
class Transformer_multi_res_7(nn.Module):
    
    def __init__(self,
                 c_in=3,
                 eval_mode=False,
                 patch_size=256,
                 padding=32,
                 overlap=32,
                 initial_stage_number=4,
                 batch_size_encoder=3,
                 batch_size_transformer=5000):
        
        super(Transformer_multi_res_7, self).__init__()
        
        self.eval_mode = eval_mode
        self.Net_first = Transformer_8(c_in=c_in,
                                       dim_hidden=[64, 128, 256, 512],
                                       eval_mode=eval_mode,
                                       batch_size_encoder=batch_size_encoder,
                                       batch_size_transformer=batch_size_transformer)
        c_in+=3
        self.Net_stage = Transformer_8(c_in=c_in,
                                       dim_hidden=[64, 128, 256, 512],
                                       eval_mode=eval_mode,
                                       batch_size_encoder=batch_size_encoder,
                                       batch_size_transformer=batch_size_transformer)
        
        self.c_in      = c_in
        self.inference_mode = False
            
        
        self.batch_size_encoder = batch_size_encoder
        self.batch_size_transformer = batch_size_transformer
        
        
        self.initial_stage_number = initial_stage_number
        self.patch_size = patch_size
        self.padding = padding
        self.overlap = overlap
        
        self.stride = self.patch_size-2*self.padding-2*self.overlap
        
        
        
        self.layer_unfold = torch.nn.Unfold(self.patch_size,
                                            dilation=1,
                                            padding=0,
                                            stride=self.stride)
           
    def set_inference_mode(self, use_cuda_eval_mode=False):
        self.inference_mode = True
        self.Net_first.set_inference_mode(use_cuda_eval_mode=use_cuda_eval_mode)
        self.Net_stage.set_inference_mode(use_cuda_eval_mode=use_cuda_eval_mode)
        
    def prepareInputs(self, x, nb_stage, stage_number):
        imgs = torch.moveaxis(x["imgs"], 2, 0)
        
            
        mask = x["mask"]
        mask = ~mask
        mask = mask.cpu()
        
        inputs = []
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape

            img   = imgs[i].contiguous().view(n * c, h * w)

            img = img.cpu()
                
            img = img.view(n, c, h, w)

            inputs.append(img)
      
        stage_number = nb_stage-stage_number
        for i in range(1, stage_number):
            
            temp = []
            for j in range(len(inputs)):

                res = decrease_size_batch(inputs[j],
                                          f=2)
                temp.append(res)
            
            inputs = temp
      
            mask = decrease_size_batch(mask,
                                       binary=True,
                                       f=2)
            
        return inputs, mask     
    
    def build_unfold(self, img):

        img1 = self.layer_unfold(img.float())
        img1 =  img1.reshape(img.shape[0],
                             img.shape[1],
                             self.patch_size,
                             self.patch_size,
                             img1.shape[-1])
     
        return img1
    
    def build_unfold_img(self, img, coord_x, coord_y):
        img = img[:,:,
                  coord_x,
                  coord_y]

        img =  img.reshape(img.shape[0],
                           img.shape[1],
                           self.patch_size,
                           self.patch_size)
     
        return img


    def build_fold(self, img, size_img, coords_x, coords_y):
        B, C, P, _, N = img.shape
        H, W = size_img
        # Flatten patches and spatial dims into one dimension: (B, C, N*P*P)
        img_flat = img.permute(0, 1, 4, 2, 3).reshape(B, C, -1)
        # Linear indices into the H×W output grid: row*W + col, shape (N*P*P,)
        flat_idx = (coords_x.permute(2, 0, 1).reshape(-1) * W +
                    coords_y.permute(2, 0, 1).reshape(-1)).long()
        flat_idx = flat_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        flat_idx = flat_idx.to(img.device)  # coords are always CPU; follow img device
        result = torch.zeros(B, C, H * W, dtype=img.dtype, device=img.device)
        result.scatter_add_(2, flat_idx, img_flat)
        return result.reshape(B, C, H, W)
    

    def find_size_stage(self, num_stage,
                        nb_stage,
                        stride=768,
                        patch_size=1024):
        
        size = int(32*(2**num_stage))
        decrease_step = nb_stage - (num_stage+1)
        return size, size, decrease_step    
    
    
    def interpolate_normal(self, normal, shape):
        normal = torch.nn.functional.interpolate(input=normal,
                                                 size=shape,
                                                 align_corners=True,
                                                 mode="bilinear")
        normal = torch.nn.functional.normalize(normal, 
                                               2, 1) 
        return normal
    
    def gen_mask_patch(self, coord_x, coord_y, shape_img):
        mask_patch = torch.ones((1, 3,
                                 self.patch_size,
                                 self.patch_size))
        border = False
        coord_x = coord_x.cpu().numpy()
        coord_y = coord_y.cpu().numpy()

        if np.min(coord_x)==0 and np.min(coord_y)!=0:
            mask_patch[:,:,
                       :-self.padding,
                       self.padding:-self.padding] = 0
            border = True
        elif np.min(coord_x)==0 and np.min(coord_y)==0:
            mask_patch[:,:,
                       :-self.padding,
                       :-self.padding] = 0
            border = True
        elif np.min(coord_x)!=0 and np.min(coord_y)==0:
            mask_patch[:,:,
                       self.padding:-self.padding,
                       :-self.padding] = 0
            border = True
            
        if np.max(coord_x)==shape_img[0]-1 and np.max(coord_y)!=shape_img[1]-1:
            mask_patch[:,:,
                       self.padding:,
                       self.padding:-self.padding] = 0
            border = True
        elif np.max(coord_x)==shape_img[0]-1 and np.max(coord_y)==shape_img[1]-1:
            mask_patch[:,:,
                       self.padding:,
                       self.padding:] = 0
            border = True
        elif np.max(coord_x)!=shape_img[0]-1 and np.max(coord_y)==shape_img[1]-1:
            mask_patch[:,:,
                       self.padding:-self.padding,
                       self.padding:] = 0
            border = True
            
        if np.min(coord_x)==0 and np.max(coord_y)==shape_img[1]-1:
            mask_patch[:,:,
                       :-self.padding,
                       self.padding:] = 0
            border = True
            
        if np.max(coord_x)==shape_img[0]-1 and np.min(coord_y)==0:
            mask_patch[:,:,
                       self.padding:,
                       :-self.padding:] = 0
            border = True
            
        if not border:
            mask_patch[:,:,
                       self.padding:-self.padding,
                       self.padding:-self.padding] = 0
        mask_patch = mask_patch
        return mask_patch
    
    def gen_weight_normal_mask(self):
        ax = np.linspace(-(self.patch_size - 1) / 2.,
                         (self.patch_size - 1) / 2.,
                         self.patch_size)
        sig = self.patch_size/5
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        weight = kernel / np.sum(kernel)
        weight = torch.from_numpy(weight)
        return weight
    
    def forward(self, x, nb_stage):
        inputs, masks = self.prepareInputs(x)
        
        pred = {}
        pred["others_scale_n"] = []
        pred["others_scale_n_error"] = []
        normal = None
        
        for i in range(nb_stage):
            normal = self.forward_stage(imgs=inputs[i],
                                        mask=masks[i],
                                        index_scale=i,
                                        normal=normal)
            stage_pred = {"n":normal}
            if i<self.nb_stage-1:
                pred["others_scale_n"].append(normal)
        
        masks[-1] = masks[-1].to(normal.device)
        pred['n'] = nn.functional.normalize(stage_pred["n"], 2, 1)
        pred['n'] = pred['n'].masked_fill(masks[-1], 0) 
        return pred
    
    
    def forward_stage(self, imgs, mask, index_scale, normal=None):
        for j in range(len(imgs)):
            if index_scale>0 and j==0: 
                normal = nn.functional.interpolate(input=normal.detach(),
                                                   size=imgs[j][0,0].shape,
                                                   align_corners=True,
                                                   mode="bilinear")
                normal = nn.functional.normalize(normal, 2, 1)

            if index_scale>0:
                normal = normal.to(imgs[j].device)
                imgs[j] = torch.cat([imgs[j], normal], 1)

        temp = torch.stack(imgs).permute(1,0,2,3,4)

        _on_cuda = next(self.Net_first.parameters()).is_cuda if index_scale == 0 \
                   else next(self.Net_stage.parameters()).is_cuda
        with torch.amp.autocast('cuda', enabled=_on_cuda):
            if index_scale==0:
                stage_pred = self.Net_first.forward([temp,
                                                     mask])
            else:
                stage_pred = self.Net_stage.forward([temp,
                                                     mask])
            
        normal = stage_pred["n"].float()  # autocast may return fp16; ensure fp32 throughout
        normal = nn.functional.normalize(normal, 2, 1)
        return normal
    
    
    
    def process(self, x, nb_stage):
        normal = None
        total_start_time = time.time()
        
        print("\n" + "="*60)
        print(f"Starting inference with {nb_stage} stages")
        print("="*60)

        for i in range(nb_stage):
            stage_start_time = time.time()
            inputs, masks = self.prepareInputs(x,
                                               nb_stage=nb_stage,
                                               stage_number=i)

            temps_imgs = []
            if i<self.initial_stage_number:
                print(f"\nStage {i+1}/{nb_stage} - Resolution: {inputs[0].shape[-1]}x{inputs[0].shape[-1]} (full image)")
                normal = self.forward_stage(imgs=inputs,
                                            mask=masks,
                                            index_scale=i,
                                            normal=normal)
                normal = normal.cpu()
                print(f"  Stage {i+1} completed in {time.time() - stage_start_time:.1f}s")

            else:
                size_stage_x, size_stage_y, decrease_step = self.find_size_stage(num_stage=i,
                                                                                 nb_stage = nb_stage,
                                                                                 stride=self.stride,
                                                                                 patch_size=self.patch_size)
   
                self.size_img_pad = (size_stage_x,
                                     size_stage_y)                
                
                normal = self.interpolate_normal(normal=normal,
                                                 shape=[size_stage_x,
                                                        size_stage_y])
                normal = torch.nn.functional.normalize(normal, 2, 1)  

                x1 = torch.arange(0, self.size_img_pad[0])
                y1 = torch.arange(0, self.size_img_pad[1])
                coords = torch.meshgrid(x1, y1,
                                        indexing='ij')
                coords_x = coords[0].unsqueeze(0).unsqueeze(0).float()
                coords_y = coords[1].unsqueeze(0).unsqueeze(0).float()
                coords_x = self.build_unfold(coords_x).long().squeeze()
                coords_y = self.build_unfold(coords_y).long().squeeze()
                
                normal_output = torch.zeros(1, 3,
                                            self.patch_size,
                                            self.patch_size,
                                            coords_x.shape[-1])

                num_patches = coords_x.shape[-1]
                print(f"\nStage {i+1}/{nb_stage} - Resolution: {size_stage_x}x{size_stage_y} ({num_patches} patches)")

                mask_weight = self.gen_weight_normal_mask()  # precompute once – same for every patch

                # ── Adaptive batch sizes for patch stages ──────────────────────
                # Patches are self.patch_size × self.patch_size (e.g. 256×256).
                # Images at this resolution are tiny compared to full-res stages,
                # so we can safely process all N images in a single encoder pass
                # and cover all spatial positions of the deepest attention layer
                # (patch_size² = 65 536 for patch_size=256) in one GPU call,
                # eliminating repeated CPU↔GPU round-trips and the O(N²) cost of
                # the growing torch.cat that was the main per-patch bottleneck.
                n_imgs = len(inputs)
                _spatial = self.patch_size * self.patch_size  # deepest stage positions
                orig_bse = self.Net_stage.batch_size_encoder
                orig_bsts = {}
                self.Net_stage.batch_size_encoder = n_imgs
                for _si in range(self.Net_stage.num_stages):
                    _pb = getattr(self.Net_stage, f"pool_block{_si + 1}")
                    orig_bsts[f"p{_si}"] = _pb.eval_mode_batch_size
                    _pb.eval_mode_batch_size = _spatial + 1
                    if _si < self.Net_stage.num_stages - 1:
                        _lb = getattr(self.Net_stage, f"light_block{_si + 1}")
                        orig_bsts[f"l{_si}"] = _lb.eval_mode_batch_size
                        _lb.eval_mode_batch_size = _spatial + 1
                # ───────────────────────────────────────────────────────────────

                # Keep Net_stage on GPU for the whole patch loop instead of
                # moving each sub-layer (~27) to GPU and back per patch call.
                # eval_mode=False uses .to(device) auto-routing so inputs are
                # moved to CUDA inside forward_stage; last_device="cuda" (set
                # by .cuda()) keeps the output on GPU until normal1.cpu() below.
                _net_was_cuda_eval = getattr(self.Net_stage, 'use_cuda_eval_mode', False)
                if _net_was_cuda_eval:
                    torch.backends.cudnn.benchmark = True  # fixed 256×256 shape repeats N times
                    self.Net_stage.change_eval_mode(eval_mode=False)
                    self.Net_stage.cuda()  # also sets last_device="cuda"
                    normal_output = normal_output.cuda()  # keep accumulator on GPU
                    mask_weight = mask_weight.cuda()      # weight tensor follows

                try:
                    for j in tqdm(range(num_patches), desc="  Processing patches", unit="patch",
                                  file=sys.stdout, dynamic_ncols=True, leave=True):
                        temps_imgs = []
                        for k in range(len(inputs)):
                            with torch.no_grad():
                                img = inputs[k]
                                img = self.build_unfold_img(img=img,
                                                            coord_x=coords_x[:, :, j],
                                                            coord_y=coords_y[:, :, j])
                                temps_imgs.append(img)

                        normal1 = self.build_unfold_img(img=normal,
                                                        coord_x=coords_x[:, :, j],
                                                        coord_y=coords_y[:, :, j])
                        with torch.no_grad():
                            mask_patches = self.gen_mask_patch(coord_x=coords_x[:, :, j],
                                                               coord_y=coords_y[:, :, j],
                                                               shape_img=self.size_img_pad)

                            mask_patches = mask_patches.to(masks.device)
                            mask_patches = (mask_patches * self.build_unfold_img(
                                img=masks,
                                coord_x=coords_x[:, :, j],
                                coord_y=coords_y[:, :, j])) > 0

                            normal1 = self.forward_stage(imgs=temps_imgs,
                                                         mask=mask_patches,
                                                         index_scale=i,
                                                         normal=normal1)

                        normal_output[:, :, :, :, j] += (normal1 * mask_weight)

                finally:
                    # Restore GPU placement and eval mode for full-res stages
                    if _net_was_cuda_eval:
                        self.Net_stage.cpu()
                        self.Net_stage.last_device = "cpu"
                        self.Net_stage.change_eval_mode(eval_mode=True, use_cuda_eval_mode=True)
                    # Restore original batch sizes for full-res stages
                    self.Net_stage.batch_size_encoder = orig_bse
                    for _si in range(self.Net_stage.num_stages):
                        getattr(self.Net_stage, f"pool_block{_si + 1}").eval_mode_batch_size = orig_bsts[f"p{_si}"]
                        if _si < self.Net_stage.num_stages - 1:
                            getattr(self.Net_stage, f"light_block{_si + 1}").eval_mode_batch_size = orig_bsts[f"l{_si}"]

                normal = self.build_fold(normal_output,
                                         coords_x = coords_x,
                                         coords_y=coords_y,
                                         size_img=self.size_img_pad)
                if _net_was_cuda_eval:
                    normal = normal.cpu()  # one transfer per stage instead of one per patch
                print(f"  Stage {i+1} completed in {time.time() - stage_start_time:.1f}s")

        total_time = time.time() - total_start_time
        print("\n" + "="*60)
        print(f"TOTAL INFERENCE TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print("="*60 + "\n")
        return {"n": normal}


    def load_weights(self,
                     file):

        checkpoint = torch.load(file, 
                                map_location=torch.device('cpu'),
                                weights_only=True)
      
        self.load_state_dict(checkpoint) 
