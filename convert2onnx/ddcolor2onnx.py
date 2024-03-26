import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
#
import torch.onnx
from onnx import numpy_helper

class ImageColorizationPipeline(object):

    def __init__(self, model_path, onnx_path, input_size=256, model_size='large'):
        
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.decoder_type = "MultiScaleColorDecoder"

        if self.decoder_type == 'MultiScaleColorDecoder':
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='MultiScaleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=100,
                num_scales=3,
                dec_layers=9,
            ).to(self.device)
        else:
            self.model = DDColor(
                encoder_name=self.encoder_name,
                decoder_name='SingleColorDecoder',
                input_size=[self.input_size, self.input_size],
                num_output_channels=2,
                last_norm='Spectral',
                do_normalize=False,
                num_queries=256,
            ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()
        
        # save onnx model
        ddcolor = self.model
        input_tensor = torch.randn(1, 3, input_size, input_size)
        torch.onnx.export(ddcolor,input_tensor,onnx_path,export_params=True,opset_version=12, input_names=["input"],output_names=["output"],verbose=False)
        print("")
        print("Done:")
        print(onnx_path)    
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrain/ddcolor_modelscope.pth')
    parser.add_argument('--onnx_path', type=str, default='pretrain/ddcolor_modelscope.onnx')
    parser.add_argument('--input_size', type=int, default=512, help='input size for model')
    parser.add_argument('--model_size', type=str, default='large', help='ddcolor model size')
    args = parser.parse_args()

    colorizer = ImageColorizationPipeline(model_path=args.model_path,onnx_path=args.onnx_path, input_size=args.input_size, model_size=args.model_size)

if __name__ == '__main__':
    main()
