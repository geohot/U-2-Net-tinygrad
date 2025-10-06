from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.helpers import get_child
import numpy as np
from PIL import Image
from skimage import io
import os
import torch
from model.u2net_tiny import U2NET, U2NETP
import time
import cv2
import argparse

def normPRED(d):
    ma, mi = d.max(), d.min()
    return (d-mi)/(ma-mi)

def save_output(image_name,predict_np,d_dir):
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+'/'+imidx+'_out.jpg')

def inference(net, input):
    # normalize the input
    tmpImg = np.zeros((input.shape[0],input.shape[1],3))
    input = input/np.max(input)

    tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
    tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

    # convert BGR to RGB
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[np.newaxis,:,:,:]
    tmpTensor = Tensor(tmpImg.astype(np.float32))

    # inference
    d1,d2,d3,d4,d5,d6,d7= net(tmpTensor)

    # normalization
    pred = 1.0 - d1[:,0,:,:]
    pred = normPRED(pred)

    # convert tinygrad tensor to numpy array
    pred = pred.squeeze()
    pred = pred.numpy()

    del d1,d2,d3,d4,d5,d6,d7

    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U^2 Net on tinygrad")

    parser.add_argument(
        "-i",
        type=str,
        default="./example_data/test2.jpg",
        help="Path to the input image"
    )

    parser.add_argument(
        "-m",
        type=str,
        default="seg",
        help="Model to load"
    )

    args = parser.parse_args()

    if args.m == "seg_small":
        unet = U2NETP(3,1)
    else:
        unet = U2NET(3,1)

    # portrait drawing model: u2net_portrait.pth"
    # human segmentation model: u2net_human_seg.pth
    print("Loading weights...")
    if args.m == "seg_small":
        loaded = torch.load("./weights/u2netp.pth", map_location="cpu")
    elif args.m == "seg":
        loaded = torch.load("./weights/u2net_human_seg.pth", map_location="cpu")
    elif args.m == "portrait":
        loaded = torch.load("./weights/u2net_portrait.pth", map_location="cpu")
    else:
        raise RuntimeError(f"Unknown model selected={args.m}")

    for k, v in loaded.items():
      get_child(unet, k).assign(v.numpy()).realize()

    image = cv2.imread(args.i)

    if args.m.startswith("seg"):
        image = cv2.resize(image, (320,320))

    print(f"Running U^2 Net on device: {Device.DEFAULT}")
    start = time.perf_counter()
    pred = inference(unet, image)
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"Inference time: {elapsed_ms:.3f} ms")

    save_output(args.i, pred, "./")
