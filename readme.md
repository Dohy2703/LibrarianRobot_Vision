# LibrarianRobot_Vision
**Seoultech university / Mechanical System Design Engineering / Capstone Design 2023** <br/>
Team : FakeSign <br/>
Vision part : <br/>
  deeplearning network + image processing (dohy2703) <br/>
  depth cam calibration + robot arm image processing (chadChang) <br/> <br/>

## Model structure
<img src="https://github.com/Dohy2703/LibrarianRobot_Vision/assets/125836071/945e3783-cd1f-447d-be5b-d45da38509f2" width="750" height="350"/>

## Test environment

NVIDIA GEFORCE RTX 3050<br/>
pytorch==2.0.1, CUDA==11.7, conda env python==3.8<br/>

## Installation

    $ cd LibrarianRobot_Vision
    $ pip install -r requirements.txt

    $ cd model/fast_demo/model/post_processing/ccl
    $ python setup.py build_ext --inplace


### ultralytics weight file
book-detection weight file : [best_v8.pt](https://drive.google.com/file/d/11x3vFYngCzosowti-MRH_S6FQLvBb-h6/view?usp=sharing)

### sample video
sample video file : [book_detection.mp4](https://drive.google.com/file/d/1wSLc7OMkNMfNMYSWZ9puQEtpamBb9p8D/view?usp=sharing) 

### fast weight file
fast weight file : [weights.pth](https://drive.google.com/file/d/12m4aaSBvcU_23w8obVT6BsfyBfM5wC_l/view?usp=sharing)

LibrarianRobot_Vision <br/>
 ㄴ fast_demo <br/>
  ㄴ model  <br/>
   ㄴ weights.pth   <-- put here !

### Text detection + recognition 


## Reference
**instance segmentation model** <br/>
ultralytics/ultralytics - https://github.com/ultralytics/ultralytics <br/>
**text detection model** <br/>
czczup/FAST - https://github.com/czczup/FAST <br/>
**text recognition model 1** <br/>
JaidedAI/EasyOCR - https://github.com/JaidedAI/EasyOCR <br/>
**text recognition model 2** <br/>
baudm/parseq - https://github.com/baudm/parseq <br/>
