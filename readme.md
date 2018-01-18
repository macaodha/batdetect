# Bat Echolocation Call Detection in Audio Recordings
Python code for the detection of bat echolocation calls in full spectrum audio recordings. This code recreate the results from the paper [Bat Detective - Deep Learning Tools for Bat Acoustic Signal Detection](https://www.biorxiv.org/content/early/2017/06/29/156869). You will also find some additional information and data on our [project page](http://visual.cs.ucl.ac.uk/pubs/batDetective).


### Training
##### 1 Download Data
Download the data from [here](http://visual.cs.ucl.ac.uk/pubs/batDetective). It contains:   
*baselines*: Results for three different commercial packages we compared against.
*models*: Pre-trained CNN models.  
*train_test_split*: the list of training and test files and the time of the bat calls in each file. The training data comes from Bat Detective and test sets have been manually verified.  
*wav*: 4,246 time expanded .wav files from the iBats project.  

##### 2 Run Training and Evaluation Code
Running *run_comparison.py* recreate the results in the paper (up to random initialization). It trains a CNN, Random Forest, and simple segmentation based models and compares their performance to three commercial systems.


### Run Detector on Your Own Data
Running *run_detector.py* loads a pre-trained CNN and performs detection on a directory of audio files. Make sure *data_dir* points to the directory where your audio files are. You need to make sure that you have a trained model on your computer. You can get one by training your own model or downloading a pre-trained one (details in the previous steps). Also make sure that if your data is already time expanded set *do_time_expansion=False*.  


### Requirements
It takes about 1.5 hrs to run on a desktop with an i7-6850K CPU, 32Gb RAM, and a GTX 1080 on Ubuntu 16.04. You might get some warnings the first time the code is run. The code has been tested with the following package versions from Conda:  
Python 2.7.12   
cython 0.24.1   
joblib 0.9.4  
lasagne 0.2.dev1    
libgcc 7.2.0   
matplotlib 2.0.2  
numpy 1.12.1  
pandas 0.19.2   
scipy 0.19.0  
scikit-image 0.13.0  
scikit-learn 0.19.0  
seaborn 0.8  
weave 0.16.0


### Video
Here is a short video that describes how our systems works.
[![Screenshot](https://img.youtube.com/vi/u35jWHdhl-8/0.jpg)](https://www.youtube.com/watch?v=u35jWHdhl-8)


### Links
[Nature Smart Cities](https://naturesmartcities.com) Deployment of smart audio detectors that use our code base to detect bats in East London.    
[Bat Detective](www.batdetective.org) Zooniverse citizen science project that was created to collected our training data.  
[iBats](http://www.bats.org.uk/pages/ibatsprogram.html) Global bat monitoring program.    


### Reference
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{batdetect17,
  title     = {Bat Detective - Deep Learning Tools for Bat Acoustic Signal Detection},
  author    = {Mac Aodha, Oisin and Gibb, Rory and Barlow, Kate and Browning, Ella and
               Firman, Michael and   Freeman, Robin and Harder, Briana and Kinsey, Libby and
               Mead, Gary and Newson, Stuart and Pandourski, Ivan and Parsons, Stuart and  
               Russ, Jon and Szodoray-Paradi, Abigel and Szodoray-Paradi, Farkas and  
               Tilova, Elena and Girolami, Mark and Brostow, Gabriel and E. Jones, Kate.},
  journal={bioRxiv},
  pages={156869},
  year={2017}
}
```

### Acknowledgements  
We are enormously grateful for the efforts and enthusiasm of the amazing iBats and Bat Detective volunteers. We would also like to thank Ian Agranat and Joe Szewczak for useful discussions and access to their systems. Finally, we would like to thank Zooniverse for setting up and hosting the Bat Detective project.

### License
Code, audio data, and annotations are available for research purposes only i.e. non-commercial use. For any other use of the software or data please contact the authors.
