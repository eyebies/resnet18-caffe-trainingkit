python split.py
caffe.bin train --solver=solver.prototxt --weights=resnet-18.caffemodel --gpu=0 2>&1 | tee output.log  
