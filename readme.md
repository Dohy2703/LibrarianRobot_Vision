# LibrarianRobot_Vision
**Seoultech university / Mechanical System Design Engineering / Capstone Design 2023** <br/>
Team : FakeSign <br/>
Vision part : <br/>
  deeplearning network + image processing (dohy2703) <br/>
  depth cam calibration + robot arm image processing (chadChang) <br/> <br/>

## Model structure
<img src="https://github.com/Dohy2703/LibrarianRobot_Vision/assets/125836071/e9994473-68cc-497a-8e2a-fb7f8329cfe4" width="750" height="350"/>

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


## Idea

### book and label paper detection



### Text detection + recognition 
<img src="https://github.com/Dohy2703/LibrarianRobot_Vision/assets/125836071/daa7b2e1-c376-45c3-8ed6-52c0648d6310" width="500" height="250"/>
I used 3 models to increase both accuracy and speed of OCR <br/>

## Reference
**instance segmentation model** <br/>
ultralytics/ultralytics - https://github.com/ultralytics/ultralytics <br/>
**text detection model** <br/>
czczup/FAST - https://github.com/czczup/FAST <br/>
**text recognition model 1** <br/>
JaidedAI/EasyOCR - https://github.com/JaidedAI/EasyOCR <br/>
**text recognition model 2** <br/>
baudm/parseq - https://github.com/baudm/parseq <br/>
