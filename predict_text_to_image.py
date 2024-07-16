import torch
import torchvision
import clip
import numpy as np
from clip import CLIP
from PIL import Image
import matplotlib.pyplot as plt
if __name__ == "__main__":
    clip = CLIP()

    image_paths = [r"D:\pythonProject\Flickr8k-clip\img\1.jpg", r"D:\pythonProject\Flickr8k-clip\img\2.jpg",
                   r"D:\pythonProject\Flickr8k-clip\img\2090545563_a4e66ec76b.jpg",
                   r"D:\pythonProject\Flickr8k-clip\img\12830823_87d2654e31.jpg"]
    captions = "A man was watching a computer"



    probs_image, probs_text = clip.detect_text(image_paths, captions)
    print("Label probs_text:", probs_text)
    print("The most fitting picture is:", image_paths[np.argmax(probs_text)])
    img = Image.open(image_paths[np.argmax(probs_text)])

    # 显示图片
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()




