from statistics import mode
import numpy as np
import cv2
import torch
import glob as glob

from torchvision import transforms
from torch.nn import functional as F
from torch import topk
from models import CNN_Model

# define computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# initialize model, switch to eval model, load trained weights
model = CNN_Model()
model = model.eval()
model.load_state_dict(torch.load('save_modelglobal_model.pth'))
print(model)
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    # print(bz, nc, h, w, sep=' ')
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name, threshold):
    vec = []
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        res = str(int(class_idx[i]))
        print('The result of classification is --', res, '--\n')
        

        cv2.imshow('CAM', result/255.)
        cv2.waitKey(0)
        cv2.imwrite(f"outputs/CAM_{save_name}.jpg", result)
        for i in range(28):
            for j in range(28):
                if result[i][j][2] >= threshold:
                    # print(i, " ", j)
                    vec.append((i, j))
    return vec

# hook the feature extractor
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('conv').register_forward_hook(hook_feature)
# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

# define the transforms, resize => tensor => normalize
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.5],
        std=[0.5])
    ])



# run for all the images in the `input` folder
for image_path in glob.glob('input/*'):
    # read the image
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    height, width, _ = orig_image.shape
    times = 3
    threshold = 253.0
    for i in range(times):
        ############################
        cv2.imshow("IMAGE", image)
        cv2.waitKey(0)
        ############################
        # apply the image transforms
        image_tensor = transform(image)
        # add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        # forward pass through model
        outputs = model(image_tensor)
        # get the softmax probabilities
        probs = F.softmax(outputs).data.squeeze()
        # get the class indices of top k probabilities
        class_idx = topk(probs, 1)[1].int()
        
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
        # file name to save the resulting CAM image with
        save_name = f"{image_path.split('/')[-1].split('.')[0]}"
        # show and save the results
        vec = show_cam(CAMs, width, height, orig_image, class_idx, save_name, threshold)
        maps = [[0 for i in range(28)] for j in range(28)]
        for (i, j) in vec:
            # print(i, " ", j)
            # print(image[i][j])
            sum = image[i][j] * 0.25
            if i - 1 >= 0 and j - 1 >= 0:
                sum += image[i - 1][j - 1] * 0.0625
            if i - 1 >= 0:
                sum += image[i - 1][j] * 0.125
            if i - 1 >= 0 and j + 1 < 28:
                sum += image[i - 1][j + 1] * 0.0625

            if i + 1 < 28 and j - 1 >= 0:
                sum += image[i + 1][j - 1] * 0.0625
            if i + 1 < 28:
                sum += image[i + 1][j] * 0.125
            if i + 1 < 28 and j + 1 < 28:
                sum += image[i + 1][j + 1] * 0.0625
            
            if j + 1 < 28:
                sum += image[i][j + 1] * 0.125
            if j - 1 >= 0:
                sum += image[i][j - 1] * 0.125

            # print(sum)
            # print(image[i][j])
        for (i, j) in vec:
            image[i][j] = maps[i][j]

        threshold -= 1