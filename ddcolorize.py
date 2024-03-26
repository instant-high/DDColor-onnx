import numpy as np
import cv2
from argparse import ArgumentParser
import onnxruntime as rt

parser = ArgumentParser()
parser.add_argument("--image", default='1.jpg', help="source image")
parser.add_argument("--output", default='1_ddcolor.jpg', help="result image")
opt = parser.parse_args()

from ddcolorizer.ddcolor import DDCOLOR

target_face = cv2.imread(opt.image)

colorizer = DDCOLOR(model_path="ddcolorizer/ddcolor_modelscope.onnx", device="cuda")

result = colorizer.process(target_face) # 512x512
#result = colorizer.process_tiny(target_face) # 256x256

cv2.imshow("Result",result.astype(np.uint8))
cv2.imwrite(opt.output,result)
cv2.waitKey() 
