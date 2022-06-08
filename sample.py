import soundfile
import MFCC

fname = 'recordings/0_jackson_0.wav' #any wav file.
data, fs = soundfile.read(fname)
mfcc = MFCC.MFCCclass(data , fs, cutpoint=12, numChannels=20)
mfcc_array = mfcc.MFCC()
print(mfcc_array)