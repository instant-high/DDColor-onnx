import cv2
import numpy as np
import onnxruntime



class DDCOLOR:
    def __init__(self, model_path="ddcolor_modelscope.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]
        


    def process(self, img):
        height, width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (512,512))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = img_gray_rgb.transpose((2, 0, 1))
        tensor_gray_rgb = np.expand_dims(tensor_gray_rgb, axis=0).astype(np.float32)

        output_ab = self.session.run(None, {(self.session.get_inputs()[0].name):tensor_gray_rgb})[0][0]
        
        output_ab = output_ab.transpose(1,2,0)

        # resize ab -> concat original l -> rgb
        output_ab_resize = cv2.resize(output_ab,(width,height)) 
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)    
        
        return output_img
        
    def process_tiny(self, img):
        height, width = img.shape[:2]

        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (256,256))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = img_gray_rgb.transpose((2, 0, 1))
        tensor_gray_rgb = np.expand_dims(tensor_gray_rgb, axis=0).astype(np.float32)

        output_ab = self.session.run(None, {(self.session.get_inputs()[0].name):tensor_gray_rgb})[0][0]
        
        output_ab = output_ab.transpose(1,2,0)

        # resize ab -> concat original l -> rgb
        output_ab_resize = cv2.resize(output_ab,(width,height))
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

        output_img = (output_bgr * 255.0).round().astype(np.uint8)    
        
        return output_img