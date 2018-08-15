# CPU Bat Detector Code

This contains python code for bat echolocation call detection in full spectrum audio recordings. This is a stripped down CPU based version of the detector with minimal dependencies that can be used for deployment.


#### Installation Instructions
* Install the Anconda Python 2.7 distribution from [here](https://www.continuum.io/downloads).
* Download this detection code from the repository and unzip it.
* Compile fast non maximum suppression by running: `python setup.py build_ext --inplace`. This might not work on all systems e.g. Windows.


#### Running on Your Own Data
* Change the `data_dir = 'wavs/'` variable so that it points to the location of the audio files you want to run the detector on.
* Specify where you want to results to be saved by setting `op_ann_dir = 'results/'`.
* To run open up the command line and type:  
  `python run_detector.py`
* If you want the detector to be less conservative in it's detections lower the value of `detection_thresh`.
* By setting `save_individual_results = False` the code will not save individual results files.

## Misc

#### Requirements
The code has been tested using Python2.7 (it mostly works under Python3.6, but we have noticed some issues) with the following package versions:  
`Python 2.7.12`   
`scipy 0.19.0`  
`numpy 1.12.1`  
`pandas 0.19.2`  
`cython 0.24.1` - not required  


#### Different Detection Models
* `detector_192K.npy` is trained to be more efficient for files that have been recorded at 192K. Note that different detectors will give different results. You can swap in your own models that have been trained using the code in `../bat_train`, and exported with '../bat_train/export_detector_weights.py'.
* To use it change the detector model as follows:  
`det_model_file = 'models/detector_192K.npy'`
* Running `evaluate_cnn_fast.py` will compute the performance of the CPU version of this CNN_FAST model on the different test sets.


#### Viewing Outputs
* The code outputs annotations as one big csv file. The location where to save the file is specified with the variable:
  `op_file_name_total = 'res/op_file.csv'`  
  It contains three fields `file_name`, `detection_time`, and `detection_prob` which indicated the time in file and detector confidence (higher is more confident) for each detected call.
* It also saves the outputs in a format compatible with [AudioTagger](https://github.com/groakat/AudioTagger). The output directory for these annotations is specified as:
`op_ann_dir = 'res/'`  
The individual `*-sceneRect.csv` files contain the same information that is specified in the main results file `op_file_name_total`, where `LabelStartTime_Seconds` corresponds to `detection_time` and `DetectorConfidence` corresponds to `detection_prob`. The additional fields (e.g. `Spec_x1`) are specific to Audiotagger and do not contain any extra information.  


#### Performance
* You can get higher resolution results by setting the `low_res` flag in `cpu_detection.run_detection()` to `False`.
* The detector code breaks the files down into chunks of audio (this is controlled by the parameter `chunk_size` in `cpu_detection` measured in seconds/10). Its best to keep this value reasonably small to keep memory usage low. However, experimenting with different values could speed things up.
* You can get faster Fourier Transform by installing FFTW3 library (http://www.fftw.org/) and python wrapper pyFFTW (https://pypi.python.org/pypi/pyFFTW). On Ubuntu Linux: `sudo apt-get install libfftw3 libfftw3-dev` and `pip install pyfftw`, respectively.



### Acknowledgements  
Thanks to Daniyar Turmukhambetov for coding help for another version of this repo. We are enormously grateful for the efforts and enthusiasm of the amazing iBats and Bat Detective volunteers. We would also like to thank Ian Agranat and Joe Szewczak for useful discussions and access to their systems. Finally, we would like to thank [Zooniverse](https://www.zooniverse.org/) for setting up and hosting the Bat Detective project.

### License
Code, audio data, and annotations are available for research purposes only i.e. non-commercial use. For any other use of the software or data please contact the authors.
