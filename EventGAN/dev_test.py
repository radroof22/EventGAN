import cv2
from tester import process_images, generate_gan_image, generate_truth_image
import numpy as np

def apply_process(img):
    alpha = 3 # Contrast control (1.0-3.0)
    beta = 00 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

if __name__ == "__main__":
    image1 = cv2.imread("/home/rohan/Pictures/Town06/00007810.png")
    image2 = cv2.imread("/home/rohan/Pictures/Town06/00007811.png")

    # image1 = apply_process(image1)
    # image2 = apply_process(image2)

    images = process_images(image1, image2)
    present_image = np.array(images.cpu()[1])
    cv2.imshow("RAW", present_image)

    # gan image
    gan_image= generate_gan_image(images)
    cv2.imshow("GAN", gan_image)

    truth_image = generate_truth_image(images)
    
    cv2.imshow("Truth", truth_image)
    cv2.waitKey(5000)