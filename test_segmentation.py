# coding:utf-8
import torch
import os
import argparse
import time
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model_TII import BiSeNet
import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os, torch
from torch.utils.data.dataset import Dataset
import PIL
# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com
import numpy as np 
from PIL import Image 
 
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('/home/wenhao/projects/SeAFusion/test_seg/SeAFusion/'+weight_name + '_' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class


class MF_dataset(Dataset):

    def __init__(self, data_dir, fuse_dir, split, input_h=480, input_w=640 ,transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted','test2'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.fuse_dir  = fuse_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image = PIL.Image.open(file_path).convert('RGB')
        image = np.asarray(image)
        return image
    
    def read_image2(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image
    
    def read_image3(self, name, folder):
        file_path = os.path.join(self.fuse_dir, '%s/%s.png' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image3(name, 'SeAFusion')
        label = self.read_image2(name, 'Label_copy') #Label_copy
        for func in self.transform:
            image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)
        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='SeAFusion')
parser.add_argument('--weight_name', '-w', type=str, default='DRDB_CAM_SIM')
parser.add_argument('--file_name', '-f', type=str, default='model_final.pth') # model_PSFusion.pth model_final.pth
parser.add_argument('--dataset_split', '-d', type=str, default='test2')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='/home/wenhao/projects/SeAFusion/output/')
parser.add_argument('--fuse_dir', '-fdr', type=str, default='/home/wenhao/projects/VIF-Benchmark/Compared_Results/test_MSRS/') # /home/wenhao/projects/VIF-Benchmark/Compared_Results/test_MSRS
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    # prepare save direcotry

    # model_dir = os.path.join('/home/wenhao/projects/segment/BANet/model/', args.weight_name)
    model_dir = os.path.join('/home/wenhao/projects/SeAFusion/model/SeAFusion_tang/')
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    conf_total = np.zeros((args.n_class, args.n_class))
    model = BiSeNet(args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)

    pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in own_state.items():
        if name not in pretrained_weight:
            print(name)
            continue
        own_state[name].copy_(param)
    print('done!')
    for name, param in pretrained_weight.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1  # do not change this parameter!
    test_dataset = MF_dataset(data_dir=args.data_dir, fuse_dir=args.fuse_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_time = time.time()
            logits,mid = model(images)
            end_time = time.time()
            if it >= 5:  # # ignore the first 5 frames
                ave_time_cost += (end_time - start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            conf_total += conf
            # save demo images
            if not os.path.exists('/home/wenhao/projects/SeAFusion/test_seg/' + args.weight_name +'/'):
                os.mkdir('/home/wenhao/projects/SeAFusion/test_seg/' + args.weight_name+'/')
            visualize(image_name=names, predictions=logits.argmax(1), weight_name='Pred_' + args.weight_name)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  % (
                      args.model_name, args.weight_name, it + 1, len(test_loader), names,
                      (end_time - start_time) * 1000))

    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

    conf_total_matfile = os.path.join('/home/wenhao/projects/SeAFusion/test_seg/' + args.weight_name, 'conf_' + args.weight_name + '.mat')
    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' % (
        args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' % (args.img_height, args.img_width))
    print('* the weight name: %s' % args.weight_name)
    print('* the file name: %s' % args.file_name)
    print(
        "* recall per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
        % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4],
           recall_per_class[5], recall_per_class[6], recall_per_class[7], recall_per_class[8]))
    print(
        "* iou per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
        % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5],
           iou_per_class[6], iou_per_class[7], iou_per_class[8]))
    print(
        "* acc per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
        % (precision_per_class[0], precision_per_class[1], precision_per_class[2], precision_per_class[3], precision_per_class[4], precision_per_class[5],
           precision_per_class[6], precision_per_class[7], precision_per_class[8]))
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f, acc: %.6f" \
          % (recall_per_class.mean(), iou_per_class.mean(), precision_per_class.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f, acc: %.6f" \
          % (np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class)), np.mean(np.nan_to_num(precision_per_class))))
    print(
        '\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' % (
            batch_size, ave_time_cost * 1000 / (len(test_loader) - 5),
            1.0 / (ave_time_cost / (len(test_loader) - 5))))  # ignore the first 10 frames
    print('\n###########################################################################')
