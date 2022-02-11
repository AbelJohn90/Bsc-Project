## REGULAR TRAINING
#../darknet/darknet detector train obj.data cfg/yolov3-tiny_3l.cfg ../pretrained/yolov3-tiny.weights -clear 1 -dont_show -mjpeg_port 8090 -map
../darknet/darknet detector train obj.data cfg/yolov3-tiny_3l.cfg -dont_show -mjpeg_port 8090 -map

## XNOR TRAINING
#../darknet/darknet detector train obj.data cfg/yolov3-tiny_3l_xnor.cfg -dont_show -mjpeg_port 8090 -map

## REGULAR TESTING
#../darknet/darknet detector map ./obj.data ./results/yolov3-tiny_3l.cfg results/best86_yuv.weights

## XNOR TESTING
#../darknet/darknet detector map ./obj.data ./cfg/yolov3-tiny_3l_xnor.cfg backup/yolov3-tiny_3l_xnor_best.weights

## Calculate anhors
#../darknet/darknet detector calc_anchors ./obj.data -num_of_clusters 9 -width 416 -height 416
