from utils.viz_utils import gen_event_images
from pytorch_utils import BaseOptions
from models.eventgan_base import EventGANBase
import configs
import cv2
import numpy as np
import torch
import os

from custom_data_event import polarize_frames, format_events

# Load Folder of Images
PATH_IMAGES = "/home/rohan/Pictures/Town04/"
image_paths = os.listdir(PATH_IMAGES)
image_paths.sort()

# Build network.
options = BaseOptions()
options.parser = configs.get_args(options.parser)
args = options.parse_args()

EventGAN = EventGANBase(args)


def process_images(prev_img, next_img):
    prev_image = cv2.resize(prev_img, (861, 260))
    next_image = cv2.resize(next_img, (861, 260))

    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    images = np.stack((prev_image, next_image)).astype(np.float32)
    images = (images / 255. - 0.5) * 2.
    images = torch.from_numpy(images).cuda()

    return images

def generate_gan_image(images):
    event_volume = EventGAN.forward(images, is_train=False)
    
    event_images = gen_event_images(event_volume[-1], 'gen')

    event_image = event_images['gen_event_time_image'][0].cpu().numpy().sum(0)

    event_image *= 255. / event_image.max()
    event_image = event_image.astype(np.uint8)
    
    return event_image

def generate_truth_image(images):
    # Get Truth Values for Event Frames
    event_truth = polarize_frames(images[0].cpu(), images[1].cpu())
    event_truth *= 255 
    event_truth = event_truth.astype(np.uint8)
    return event_truth

def apply_process(img):
    alpha = 1.75 # Contrast control (1.0-3.0)
    beta = 20 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

if __name__ == "__main__":
    
    for i in range(500, len(image_paths)): 
        # Read in images.
        prev_image_raw = cv2.imread(PATH_IMAGES+image_paths[i-1])
        next_image_raw = cv2.imread(PATH_IMAGES+image_paths[i])

        prev_image, next_image = apply_process(prev_image_raw), apply_process(next_image_raw)
        images = process_images(prev_image, next_image)
        

        event_image = generate_gan_image(images)
        
        event_truth = generate_truth_image(images)

        # Handle displaying live video of the images being simulated
        cv2.imshow("GAN", event_image)
        cv2.imshow("TRUTH", event_truth)
        cv2.imshow("RAW", next_image_raw)
        cv2.waitKey(1)