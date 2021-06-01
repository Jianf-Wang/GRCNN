_base_ = './mask_rcnn_grcnn55_fpn_2x_coco.py'
model = dict(pretrained='./checkpoint_params_skgrcnn109.pt', backbone=dict(name='SKGRCNN109'))
