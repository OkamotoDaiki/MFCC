# MFCC (Mel-frequency cepstrum coefficients)
 
Class that extracts features of voice data by MFCC using a mel-filter bank.
 
# DEMO
 
This repository contains the generation of mel filter banks and the output of MFCC. The figure of the mel filter bank when the frequency parameter(fo) is 700Hz is shown below.<br>

![mel-filterbank_fo=700Hz](https://user-images.githubusercontent.com/49944765/173179132-a357ea67-5c45-47e4-84e9-ea1489ff2d5b.png)

"""Machine Learning TEST"""<br>
The accuracy was `84.5%` when the dataset of audio sample was as follows and output to one linear SVM of the machine learning algorithm using the MFCC output by this program.

* Name:  Jakobovski / Free Spoken Digit Dataset (FSDD)
* LICENCE: Creative Commons Attribution-ShareAlike 4.0 International
* Link: https://github.com/Jakobovski/free-spoken-digit-dataset

Please refer to the following repository for the above results.
* Link: https://github.com/OkamotoDaiki/MFCC_linearSVM_TEST

# Features
 
Modified a possible bug and modified it to a class that anyone can handle.
It can be used by making the MFCC class of the MFCC module an object and calling the MFCC() function. At least a time series data array and a sampling rate are required.

 
# Requirement

* Python 3.8.10
* numpy 1.21.4
 
# Installation
 
You can import it with the following program.
 
```python
import soundfile
import MFCC

fname = 'recordings/0_jackson_0.wav' #any wav file.
data, fs = soundfile.read(fname)
mfcc = MFCC.MFCCclass(data , fs, cutpoint=12, numChannels=20)
mfcc_array = mfcc.MFCC()
```
 
# Usage
 
Refer to the sample program.

# Author
* Oka.D.
* okamotoschool2018@gmail.com