# Bat Echolocation Call Detection in Audio Recordings
Python code for the detection of bat echolocation calls in full spectrum audio recordings. This code recreate the results from the paper [Bat Detective - Deep Learning Tools for Bat Acoustic Signal Detection](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005995). You will also find some additional information and data on our [project page](http://visual.cs.ucl.ac.uk/pubs/batDetective).


**Update Dec 2022:** We now have a new and improved codebase that you can access [here](https://github.com/macaodha/batdetect2).  


## Training
`bat_train` contains the code to train the models and recreate the plots in the paper.

## Running the Detector
`bat_eval` contains lightweight python scripts that load a pretrained model and run the detector on a directory of audio files. No GPU is required for this step.  


## Misc

#### Video
Here is a short video that describes how our systems works.  
[![Screenshot](https://img.youtube.com/vi/u35jWHdhl-8/0.jpg)](https://www.youtube.com/watch?v=u35jWHdhl-8)


#### Links
[Nature Smart Cities](https://naturesmartcities.com) Deployment of smart audio detectors that use our code base to detect bats in East London.    
[Bat Detective](www.batdetective.org) Zooniverse citizen science project that was created to collected our training data.  
[iBats](http://www.bats.org.uk/pages/ibatsprogram.html) Global bat monitoring program.    


#### Reference
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{batdetect18,
  title     = {Bat Detective - Deep Learning Tools for Bat Acoustic Signal Detection},
  author    = {Mac Aodha, Oisin and Gibb, Rory and Barlow, Kate and Browning, Ella and
               Firman, Michael and   Freeman, Robin and Harder, Briana and Kinsey, Libby and
               Mead, Gary and Newson, Stuart and Pandourski, Ivan and Parsons, Stuart and  
               Russ, Jon and Szodoray-Paradi, Abigel and Szodoray-Paradi, Farkas and  
               Tilova, Elena and Girolami, Mark and Brostow, Gabriel and E. Jones, Kate.},
  journal={PLOS Computational Biology},
  year={2018}
}
```

#### Acknowledgements  
We are enormously grateful for the efforts and enthusiasm of the amazing iBats and Bat Detective volunteers. We would also like to thank Ian Agranat and Joe Szewczak for useful discussions and access to their systems. Finally, we would like to thank [Zooniverse](https://www.zooniverse.org/) for setting up and hosting the Bat Detective project.
