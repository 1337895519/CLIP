from PIL import Image
import numpy as np
from clip import CLIP

if __name__ == "__main__":
    clip = CLIP()
    
    # 图片的路径
    image_path = r"D:\pythonProject\Flickr8k-clip\datasets\flickr8k-images\3346918203_986dca6641.jpg"
    # 寻找对应的文本，4选1
    captions   = [
        "A bicyclist in blue goes up a hill by the woods .",
        "A person is riding their bike on a trail next to the woods .",
        "A woman rides her bike up a hill near the woods .",
        "A woman in a blue jacket is riding a bicycle on a woodland path .",
        "A woman mountain biker with a backpack bikes up a hill ."
    ]
    
    image = Image.open(image_path)
    probs_image,probs_text = clip.detect_image(image, captions)
    print("Label probs:", probs_image)
    print("Label:", captions[np.argmax(probs_image)])