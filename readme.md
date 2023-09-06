# LibrarianRobot_Vision
**Seoultech university / Mechanical System Design Engineering / Capstone Design 2023** <br/>
Team : FakeSign <br/>
Vision part : <br/>
  deeplearning network + image processing (dohy2703) <br/>
  depth cam calibration + robot arm image processing (chadChang) <br/> <br/>

## model structure
<img src="https://github.com/Dohy2703/LibrarianRobot_Vision/assets/125836071/945e3783-cd1f-447d-be5b-d45da38509f2" width="750" height="350"/>


## installation

    $ cd LibrarianRobot_Vision <br/>
    $ pip install -r requirements.txt <br/>

    $ cd model/fast_demo/model/post_processing/ccl<br/>
    $ python setup.py build_ext --inplace <br/> <br/>


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
