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

def load_model(cfgfile, weightfile,verbose=False):
    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    if verbose:
        model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    return model

def valid(model, datafolder, datacfg, inputimage, shape=None):

    # Parse configuration files
    options      = read_data_cfg(datacfg)
    meshname     = datafolder + options['mesh']
    name         = options['name']
    modelsize    = np.float32(options['diam'])

    # Parameters
    seed         = int(time.time())
    gpus         = '0'     # Specify which gpus to use
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
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

    #logging("   Testing {}...".format(name))

    if isinstance(inputimage, str):
        image = Image.open(inputimage).convert('RGB')
        if shape is not None:
            image = image.resize(shape)
    else:
        # to avoid the error:
        # some of the strides of a given numpy array are negative. This is currently not supported, but will be added in future releases.
        # ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        image = Image.fromarray(inputimage)
        if shape is not None:
            image = image.resize(shape)
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
    #boxes   = all_boxes[0]

    best_conf_est = -1

    # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
    ##
    #
    # boxes[j] sono relative ad una sola detection
    # il ciclo for sulla confidenza e' per scegliere solo
    # la detection piu' confidente di questo oggetto
    #
    ##
    for i in range(len(all_boxes)):
        boxes = all_boxes[i]
        for j in range(len(boxes)):
            if (boxes[j][18] > best_conf_est):
                box_pr        = boxes[j]
                best_conf_est = boxes[j][18]

    # Denormalize the corner predictions 
    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480
    preds_corners2D.append(corners2D_pr)
    
    # Compute [R|t] by pnp
    points3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
    R_pr, t_pr = pnp(points3D, corners2D_pr, internal_calibration.astype(np.float32))

    return R_pr, t_pr, all_boxes, modelsize, best_conf_est

if __name__ == '__main__':

    datafolder = '../DATA/linemod/'
    cname = 'cup'

    modelweights = datafolder + 'backup/' + cname
    if os.path.exists(modelweights + '/model.weights'):
        modelweights = modelweights + '/model.weights'
    else:
        modelweights = modelweights + '/model_backup.weights'
    model = load_model('cfg/yolo-pose.cfg', modelweights, verbose=False)

    for i in range(10):
        imagefilename = datafolder + 'LINEMOD/' + cname + '/JPEGImages/' + str(np.random.randint(0,1200)).zfill(6) + '.jpg'
        R, t = valid(model, datafolder, 'cfg/' + cname + '.data', imagefilename)
