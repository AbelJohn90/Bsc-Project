from gga.src.CSM import cosine_similarity_maps as cs_map
from yolo.dataset_gen.utils import load_checkpoint, get_loaders
from tool import darknet2pytorch
from saliency_map_generator as sal_gen

def main():
    '''
    converts yolo3 format .weights to pytorch format .pth

    stores it in yolo/results/
    '''

    # load weights from darknet format
    model = darknet2pytorch.Darknet('yolo/results/yolov3-tiny_3l.cfg', inference=True)
    model.load_weights('yolo/results/best86_yuv.weights')

    '''
    we now want to get the output of all the images and their respective tensor, and put them all
    in one big tensor such that we can use the cosine_similarity_maps
    '''

    # 
    train_loader, test_loader, eval_loader = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    X = torch.Tensor()
    for x in eval_loader:
        X.append(x)

    cs_map(model, )



