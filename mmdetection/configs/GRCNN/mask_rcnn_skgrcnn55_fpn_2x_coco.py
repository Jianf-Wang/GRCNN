_base_ = './mask_rcnn_grcnn55_fpn_2x_coco.py'
model = dict(pretrained='./checkpoint_params_skgrcnn55.pt', backbone=dict(name='SKGRCNN55'))
