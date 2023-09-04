import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

try:
    from model import FAST
    from model.utils import fuse_module, rep_model_convert
    from prepare_input import process_image, scale_aligned_short
    from config.fast.tt import fast_base_tt_512_finetune_ic17mlt
except:
    from fast_demo.model import FAST
    from fast_demo.model.utils import fuse_module, rep_model_convert
    from fast_demo.prepare_input import process_image, scale_aligned_short
    from fast_demo.config.fast.tt import fast_base_tt_512_finetune_ic17mlt
import torch
import cv2
import numpy as np
import time

from parseq.strhub.data.module import SceneTextDataModule
from PIL import Image
import torchvision.transforms as transforms

try:
    import EasyOCR.easyocr.easyocr as easyocr
except:
    import easyocr  # pip install easyocr


def setup_fast():
    cfg = fast_base_tt_512_finetune_ic17mlt

    model = FAST()
    model = model.cuda()
    try:
        checkpoint = torch.load('model/weights.pth')
    except :
        checkpoint = torch.load('/home/kdh/PycharmProjects/pythonProject4yolov8/fast_demo/model/weights.pth')

    state_dict = checkpoint['ema']
    d = dict()
    for key, value in state_dict.items():
        tmp = key.replace("module.", "")
        d[tmp] = value

    model.load_state_dict(d)
    model = rep_model_convert(model)
    model = fuse_module(model)
    model.eval()

    return model, cfg

def FAST_crop(image, model, cfg, visualize=False, count_time=True, show_img = True):
    if count_time:
        start_t = time.time()

    if isinstance(image, str)==True:
        if image.endswith('.png') or image.endswith('.jpg'): # if path
            img = cv2.imread(image)
            image = process_image(image)
        else:
            raise Exception('Image file should be .png or .jpg file')
    elif isinstance(image, np.ndarray):
        img = image.copy()
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

    image['imgs']= image['imgs'].unsqueeze(0).cuda(non_blocking=True)

    with torch.no_grad():
        outputs = model(**image, cfg=cfg)
        result = outputs['results'][0]['bboxes']

    crop_img = []
    i=0
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
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

        crop_img.append(img[y:y + h, x:x + w])
        # i += 1
        # cv2.imwrite('/home/kdh/Desktop/CapstoneDesign_DeepLearning/crop/FAST/'+str(i)+'.png', img[y:y + h, x:x + w])
        if visualize:
            cv2.imshow('crop_img', img[y:y + h, x:x + w])
            cv2.waitKey(0)

    if count_time:
        end_t = time.time()
        print('reference time : ', end_t - start_t)

    if visualize or show_img:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    return crop_img


def read_text_easyocr(reader, crop_img):
    exact_list = []

    for item in crop_img:
        result = reader.recognize(item)
        for i in range(len(result)):
            # print(result[i][1])
            if 'E' in result[i][1] or 'M' in result[i][1] or 'm' in result[i][1] or result[i][1][-6:].isdigit==True:
                exact_list.append(item)

    if len(exact_list) == 0:
        return crop_img
    return exact_list

    # reader = easyocr.Reader(['en'])  # 한글도 적용하고 싶으면 ['ko', 'en']. 다만 여기선 자음만 인식 안돼서 안씀
    # result = reader.readtext(sharpening_img, slope_ths=0.3)


def set_parseq():
    # Load model and image transforms
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    return parseq, img_transform

def parseq_recognize(image, parseq, img_transform):
    if isinstance(image, str) and ( image.endswith('.png') or image.endswith('.jpg') ):
        img = Image.open(image).convert('RGB')
        img = img_transform(img).unsqueeze(0) # (B, C, H, W)

        logits = parseq(img)
        logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

        # Greedy decoding
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        print('Decoded label = {}'.format(label[0]))

    elif isinstance(image, list):
        recog_list = []

        for item in image:
            # img = Image.open(item).convert('RGB')
            img = Image.fromarray(np.uint8(item)).convert('RGB')
            img = img_transform(img).unsqueeze(0)  # (B, C, H, W)

            logits = parseq(img)
            logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

            # Greedy decoding
            pred = logits.softmax(-1)
            label, confidence = parseq.tokenizer.decode(pred)
            # print('Decoded label = {}'.format(label[0]))
            recog_list.append(label[0])

        return recog_list


if __name__ == '__main__':
    # image_path = '/home/kdh/Desktop/realsense_save/20_1/03.png'
    image_path = '/home/kdh/Desktop/realsense_save/20_1/16.png'
    img = cv2.imread(image_path)

    model, cfg = setup_fast()
    parseq, img_transform = set_parseq()
    reader = easyocr.Reader(['en'])

    start_t = time.time()
    # fast_img = FAST_crop(image=image_path, model=model, cfg=cfg, visualize=False, show_img=False, count_time=False)
    fast_img = FAST_crop(image=img, model=model, cfg=cfg, visualize=False, show_img=False, count_time=False)
    exact_list = read_text_easyocr(reader, fast_img)
    recog_list = parseq_recognize(exact_list, parseq, img_transform)

    end_t = time.time()
    print('infer time : ', end_t - start_t)


    # ij = 0
    # for i in fast_img:
    #     cv2.imshow(str(ij), i)
    #     cv2.waitKey(0)