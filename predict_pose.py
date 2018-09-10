import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
warnings.filterwarnings("ignore")

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly

def load_model(cfgfile, weightfile):
    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    return model

def valid(model, datafolder, datacfg, imagefilename):

    # Parse configuration files
    options      = read_data_cfg(datacfg)
    meshname     = datafolder + options['mesh']
    name         = options['name']

    # Parameters
    seed         = int(time.time())
    gpus         = '0'     # Specify which gpus to use
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    use_cuda        = True
    num_classes     = 1
    conf_thresh     = 0.1

    # To save
    preds_corners2D     = []

    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D     = get_3D_corners(vertices)

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()

    logging("   Testing {}...".format(name))

    image = Image.open(imagefilename).convert('RGB')
    transf = transforms.Compose([transforms.ToTensor(),])
    data = transf(image)
    data.unsqueeze_(0)
    
    # Pass data to GPU
    if use_cuda:
        data = data.cuda()
    
    # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
    data = Variable(data, volatile=True)
    
    # Forward pass
    output = model(data).data
    
    # Using confidence threshold, eliminate low-confidence predictions
    all_boxes = get_region_boxes(output, conf_thresh, num_classes)
    
    # Get all the predictions
    boxes   = all_boxes[0]

    best_conf_est = -1

    # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
    for j in range(len(boxes)):
        if (boxes[j][18] > best_conf_est):
            box_pr        = boxes[j]

    # Denormalize the corner predictions 
    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480
    preds_corners2D.append(corners2D_pr)
    
    # Compute [R|t] by pnp
    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

    def draw_axis(img, R, t, K):
        import cv2
        # unit is mm
        rotV, _ = cv2.Rodrigues(R)
        points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
        return img
    
    output_frame = draw_axis(np.asarray(image), R_pr, t_pr, internal_calibration)

    import matplotlib.pyplot as plt
    plt.imshow(output_frame)
    plt.show()

    return R_pr, t_pr

if __name__ == '__main__':

    datafolder = '../DATA/linemod/'
    cname = 'ape'
    imagefilename = datafolder + 'LINEMOD/' + cname + '/JPEGImages/000000.jpg'

    model = load_model('cfg/yolo-pose.cfg', datafolder + 'backup/' + cname + '/model_backup.weights')
    R, t = valid(model, datafolder, 'cfg/' + cname + '.data', imagefilename)
    logging("    R: {}".format(R))
    logging("    t: {}".format(t))
