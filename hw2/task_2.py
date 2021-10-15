from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL
from utils import *
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders

train_dataset = VOCDataset('trainval', 512, top_n=300)
val_dataset = VOCDataset('test', 512, top_n=300)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)

own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue

# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
for name, param in net.named_parameters():
    if "features.0" in name:
        param.requires_grad = False

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fn, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 10
val_interval = 1000


def test_net(model, val_loader=None, thresh=0.05, epoch_num=1):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """
    counter = 0
    tp_all, num_pos_all, num_gt_all = np.zeros((20, 1)), np.zeros((20, 1)), np.zeros((20, 1))

    for iter, data in enumerate(val_loader):
        # one batch = data for one image
        image           = data['image'].cuda()
        target          = data['label'].cuda()
        wgt             = data['wgt'].cuda()
        rois            = data['rois'].cuda()
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        for i in range(len(gt_class_list)):
            num_gt_all[gt_class_list[i].item(), 0] += 1

        # TODO: perform forward pass, compute cls_probs
        cls_probs = model(image, rois * 512, target)

        bboxes_all_classes = []
        classes_all_boxes = []
        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):
            # get valid rois and cls_scores based on thresh
            probs = cls_probs[:, class_num]
            bboxes = rois[0] * 512
            # use NMS to get boxes and scores
            if torch.max(probs) < thresh:
                continue
            pred_bboxes, pred_scores = nms(bboxes, probs, thresh)
            num_pos_all[class_num, 0] += len(pred_bboxes)

            matched = []
            for box in pred_bboxes:
                bboxes_all_classes.append(box / 512)
                classes_all_boxes.append(class_num)
                for i, gt_box in enumerate(gt_boxes):
                    if gt_class_list[i].item() == class_num and i not in matched:
                        if iou(box / 512, gt_box) > 0.3:
                            tp_all[class_num, 0] += 1
                            matched.append(i)
                            break

        # TODO: visualize bounding box predictions when required
        # # visualize the first 10 images
        # if counter < 10:
        #     class_id_to_label = dict(enumerate(val_dataset.CLASS_NAMES))
        #     img = wandb.Image(image, boxes={
        #         "predictions": {
        #             "box_data": get_box_data(classes_all_boxes, bboxes_all_classes),
        #             "class_labels": class_id_to_label,
        #         },
        #     })
        #     wandb.log({"Epoch" + str(epoch_num) + "_" + "image" + str(counter+1): img})
        #     counter += 1
        # TODO: Calculate mAP on test set

    AP = (tp_all / num_pos_all) * (tp_all / num_gt_all)
    wandb.log({"map": np.mean(AP)})
    return AP


USE_WANDB = True
if USE_WANDB:
    wandb.init(project="vlr2", reinit=True)


AP_all = np.zeros((20, 5))
for i in range(5):
    for iter, data in enumerate(train_loader):
        # TODO: get one batch and perform forward pass
        # one batch = data for one image
        image           = data['image'].cuda()
        target          = data['label'].cuda()
        wgt             = data['wgt'].cuda()
        rois            = data['rois'].cuda()
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        # TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
        # also convert inputs to cuda if training on GPU
        net_output = net(image, rois * 512, target)

        # backward pass and update
        loss = net.loss
        train_loss += loss.item()
        step_cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_cnt % 100 == 0:
            print(step_cnt, train_loss / 500)
            wandb.log({"train_loss": train_loss / 500})
            train_loss = 0.

    # TODO: evaluate the model every N iterations (N defined in handout)
    # if iter % val_interval == 0 and iter != 0:
    # if i == 0 or i == 4:
    net.eval()
    ap = test_net(net, val_loader, thresh=0.05, epoch_num=i)
    print("AP ", ap)
    for j in range(20):
        wandb.log({"class" + str(j): ap[j, 0]})
    net.train()
    # TODO: Perform all visualizations here
    # The intervals for different things are defined in the handout
