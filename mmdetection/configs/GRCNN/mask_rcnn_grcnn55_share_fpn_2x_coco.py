_base_ = './mask_rcnn_grcnn55_fpn_2x_coco.py'
model = dict(pretrained='./checkpoint_params_grcnn55_weight_share.pt', backbone=dict(name='GRCNN55_SHARE'))
