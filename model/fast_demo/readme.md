
**FAST** <br/> 
FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation <br/><br/>
**Thanks to** <br/>
  https://github.com/czczup/FAST<br/>
  https://github.com/Chappie74/fast_demo<br/><br/>

custom data inference code<br/> <br/>


**Installation & Setting** <br/>

    $ pip install -r requirements.txt

then,<br/>

    $ cd model/post_processing/ccl
    $ python setup.py build_ext --inplace
    

<br/>

**Change log**<br/>
fast_head.py <br/>

    line6 | from ..utils import generate_bbox --> from ..utils.generate_bbox import generate_bbox

prepare_input.py <br/>

    line47 | org_img_size=np.array(img.shape[:2]) --> org_img_size=[np.array(img.shape[:2])]
    line52 | img_size=np.array(img.shape[:2]), --> img_size=[np.array(img.shape[:2])],

generate_bbox.py <br/>

    line15 | if points.shape[0] < cfg.test_cfg.min_area : --> if points.shape[0] < cfg.test_cfg['min_area']:
    line19 | if score_i < cfg.test_cfg.min_score : --> if score_i < cfg.test_cfg['min_score']:
    line23 | if cfg.test_cfg.bbox_type == 'rect': --> if cfg.test_cfg['bbox_type'] == 'rect':
    line29 | elif cfg.test_cfg.bbox_type == 'poly': --> elif cfg.test_cfg['bbox_type'] == 'poly':

fast_neck.py <br/>

    line31 | return F.upsample(x, size=(H, W), mode='bilinear') --> return F.interpolate(x, size=(H, W), mode='bilinear')

<br/>

**Inference custom data**<br/>

test.py <br/>

    line31 | image_path = ' ' <-- your image path
