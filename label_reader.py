from model.fast_demo.model import FAST
from model.fast_demo.model.utils import fuse_module, rep_model_convert
from model.fast_demo.prepare_input import process_image, scale_aligned_short
from model.fast_demo.config.fast.tt import fast_base_tt_512_finetune_ic17mlt
from model.parseq.strhub.data.module import SceneTextDataModule

import torch
import cv2
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms
import easyocr

class labelReader:
    def __init__(self):
        self.fast_model = None
        self.fast_cfg = None
        self.easyocr_reader = None
        self.parseq = None

        self.img_transform = None
        self.crop_image = []
        self.exact_list = []
        self.np_image = np.array([])

    def setup_reader(self, cfg=fast_base_tt_512_finetune_ic17mlt):
        # fast
        fast_model = FAST().cuda()
        checkpoint = torch.load('./model/fast_demo/model/weights.pth')
        state_dict = checkpoint['ema']
        d = dict()
        for key, value in state_dict.items():
            tmp = key.replace("module.", "")
            d[tmp] = value
        fast_model.load_state_dict(d)
        fast_model = rep_model_convert(fast_model)
        fast_model = fuse_module(fast_model)
        fast_model.eval()

        self.fast_model = fast_model
        self.fast_cfg = cfg

        # easyocr
        self.easyocr_reader = easyocr.Reader(['en'])

        # parseq
        self.parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        self.img_transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)

    def FAST_crop(self, image, visualize=False, count_time=False, show_img = False):
        org_image = None

        if count_time:
            start_t = time.time()

        if isinstance(image, str)==True:  # 주소일 때
            if image.endswith('.png') or image.endswith('.jpg'): # if path
                org_image = cv2.imread(image)
                image = process_image(image)
            else:
                raise Exception('Image file should be .png or .jpg file')
        elif isinstance(image, np.ndarray):  # 넘파이 행렬일 때
            org_image = image.copy()
            image = image[:, :, [2, 1, 0]]
            img_meta = dict(
                org_img_size=[np.array(image.shape[:2])]
            )
            image = scale_aligned_short(image)
            img_meta.update(dict(
                img_size=[np.array(image.shape[:2])],
                filename='00'
            ))

            image = Image.fromarray(image)
            image = image.convert('RGB')
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

            image = dict(
                imgs=image,
                img_metas=img_meta
            )

        self.np_image = org_image
        image['imgs']= image['imgs'].unsqueeze(0).cuda(non_blocking=True)

        with torch.no_grad():
            outputs = self.fast_model(**image, cfg=self.fast_cfg)
            result = outputs['results'][0]['bboxes']

        crop_img = []
        for bbox in result:
            points = np.array(bbox).reshape(-1,2)

            ''' visualize points '''
            # if visualize:
            #     for pt in points:
            #         cv2.circle(img, pt, 1, (0, 255, 0), 2)
            ''' visualize minarearect '''
            # if visualize:
            #     rect = cv2.minAreaRect(points)
            #     box = cv2.boxPoints(rect)
            #     box = np.intp(box)
            #     cv2.drawContours(img, [box], 0, (0,255,0), 3)
            ''' visualize bbox '''
            x, y, w, h = cv2.boundingRect(points)
            if visualize:
                cv2.rectangle(org_image, (x, y), (x+w, y+h), (0, 0, 255), 3)

            crop_img.append([org_image[y:y + h, x:x + w], x, y, x+w, y+h])
            if visualize:
                cv2.imshow('crop_img', org_image[y:y + h, x:x + w])
                cv2.waitKey(0)

        if count_time:
            end_t = time.time()
            print('reference time : ', end_t - start_t)

        if visualize or show_img:
            cv2.imshow('img', org_image)
            cv2.waitKey(0)

        self.crop_img = crop_img

    def read_text_easyocr(self):
        exact_list = []
        for item in self.crop_img:
            result = self.easyocr_reader.recognize(item[0])
            for i in range(len(result)):
                # print(result[i][1])
                if 'E' in result[i][1] or 'M' in result[i][1] or 'm' in result[i][1] or result[i][1][
                                                                                        -6:].isdigit == True:
                    exact_list.append(item)

        if len(exact_list) == 0:
            self.exact_list = self.crop_img
        else:
            self.exact_list = exact_list

    def parseq_recognize(self):
        recog_list = []

        for item in self.exact_list:
            img = Image.fromarray(np.uint8(item[0])).convert('RGB')
            img = self.img_transform(img).unsqueeze(0)  # (B, C, H, W)

            logits = self.parseq(img)
            logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

            # Greedy decoding
            pred = logits.softmax(-1)
            label, confidence = self.parseq.tokenizer.decode(pred)
            # print('Decoded label = {}'.format(label[0]))
            recog_list.append([label[0], item[1], item[2], item[3], item[4]])

        return recog_list

if __name__ == '__main__':
    image_path = '/home/kdh/Desktop/realsense_save/20_1/03.png'
    # image_path = '/home/kdh/Desktop/realsense_save/20_1/16.png'
    img = cv2.imread(image_path)


    RD = labelReader()
    RD.setup_reader()

    start_t = time.time()
    RD.FAST_crop(img)
    RD.read_text_easyocr()
    recog_list = RD.parseq_recognize()

    print(recog_list)

    end_t = time.time()
    print('infer time : ', end_t - start_t)

