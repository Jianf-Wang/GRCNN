python imagenet_train.py \
  --epochs 120 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
#  --resume checkpoint.pth.tar \
   
