import glob
import soundfile
import FFTTool
import MFCC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


if __name__=="__main__":
    fpath = "recordings/*.wav"
    files = glob.glob(fpath)
    training_list = []
    label_list = []

    for fname in files:
        data, fs = soundfile.read(fname)
        data = FFTTool.ZeroPadding(data).process()
        window_data = np.hamming(len(data)) * data
        numChannels = 20
        cutpoint = 12
        mfcc = MFCC.MFCCclass(window_data , fs, cutpoint=12, numChannels=20)
        mfcc_array = mfcc.MFCC()
        training_list.append(mfcc_array)
        label = fname.split('/')[1].split('_')[0]
        label_list.append(label)


    X_train, X_test, y_train, y_test = train_test_split(
        training_list, label_list, random_state=0
    )

    Linear_svm = LinearSVC().fit(X_train, y_train)
    score_testset = Linear_svm.score(X_test, y_test)
    print("Accuracy on test set: {}".format(score_testset))
