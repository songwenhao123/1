# coding:utf-8
import os
import argparse
from utils_old import *
import torch
from torch.utils.data import DataLoader
from dataset import Fusion_dataset2
from FusionNet import FusionNet, Fusion_DRDB, Baseline, Fusion_DRDB_CAM, DRDB_LFE,DRDB_CAM_SIM,WO_DRDB,WO_CAM, R1_DRDB, WO_SIM
from model_TII import BiSeNet
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
import time

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()
# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
def main(ir_dir='./test_imgs/ir', vi_dir='./test_imgs/vi',mask_dir='./test_imgs/mask', save_dir='./SeAFusion', fusion_model_path='./model/Fusion/fusionmodel_final.pth', Seg_model_path='./model/Fusion/fusionmodel_final.pth'):
    # fusionmodel = WO_CAM(output=1, addition_mode='l1_norm')
    x = torch.randn(1, 1, 640, 480).cuda()
    y = torch.randn(1, 1, 640, 480).cuda()
    z = torch.randn(1, 9, 640, 480).cuda()
    fusionmodel = DRDB_CAM_SIM(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel = fusionmodel.to(device)
    print('fusionmodel load done!')
    print("Params(M): %.3f" % (params_count(fusionmodel) / (1000 ** 2)))
    # flops = FlopCountAnalysis(fusionmodel, (x, y,z))
    # print("FLOPs(G): %.3f" % (flops.total()/1e9))
    n_classes = 9
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(Seg_model_path))
    # torch.save(net.state_dict(), save_pth, file_serialization=False)
    net.to(device)
    net.eval()
    print("SEGParams(M): %.3f" % (params_count(net) / (1000 ** 2)))

    test_dataset = Fusion_dataset2('val', ir_path=ir_dir, vi_path=vi_dir, mask_path=mask_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    result = []
    total_time = 0 
    with torch.no_grad():
        for it, (img_vis, img_ir,img_mask, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            img_mask = img_mask.to(device)
            out, mid = net(img_mask)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            torch.cuda.synchronize()
            st = time.time()
            fused_img = fusionmodel(vi_Y, img_ir, out)
            torch.cuda.synchronize()
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))
            elapsed_time = time.time() - st
            result.append(elapsed_time)
            total_time += elapsed_time 
        avg_time = np.mean(result)
        print("Avg Time: {:.3f}s\n".format(avg_time))
        print("Total Time: {:.3f}s\n".format(total_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--model_path', '-M', type=str, default='/home/wenhao/projects/SeAFusion/R2_model/SFINet/fusion_model.pth')
    parser.add_argument('--Seg_model_path', '-S', type=str, default='/home/wenhao/projects/SeAFusion/R2_model/SFINet/model_final.pth')
    ## dataset
    parser.add_argument('--ir_dir', '-ir_dir', type=str, default='/shares/image_fusion/IVIF_datasets/test/test_MSRS/ir')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, default='/shares/image_fusion/IVIF_datasets/test/test_MSRS/vi')
    parser.add_argument('--mask_dir', '-mask_dir', type=str, default='/shares/image_fusion/IVIF_datasets/test/test_MSRS/vi')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='/home/wenhao/projects/SeAFusion/output/R2/MSRS')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('WO_CAM_avg', args.gpu))
    main(ir_dir=args.ir_dir, vi_dir=args.vi_dir, mask_dir=args.mask_dir, save_dir=args.save_dir, fusion_model_path=args.model_path, Seg_model_path=args.Seg_model_path)
