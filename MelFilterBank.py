import numpy as np
import warnings
warnings.simplefilter('ignore')

class MelFilterBankCalc:
    """
    Class for calculate mel-filterbank.
    To calculate, call the melfilterbank fucntion.

    Attributes:
        N: Data length.
        fs: samplingrate
        fo: frequency Parameter. Default setting is 700.
        mel: Definition of the mel scale. Default setting is 1000.
        numChannels: Specifying the number of bandpass fileters. Default setting is 20.
    """
    def __init__(self, N, fs, fo=700, mel=1000, numChannels=20):
        self.N = N
        self.fs = fs
        self.fo = fo
        self.mel = mel
        self.numChannels = numChannels


    def calc_mo(self):
        """
        Functions for determining dependent parameters of the Mel scale.
        """
        return self.mel / np.log((self.mel / self.fo) + 1.0)


    def hz2mel(self, f):
        """
        Convert Hz to mel.
        """
        mo = self.calc_mo()
        return mo * np.log(f / self.fo + 1.0)
    

    def mel2hz(self, m):
        """
        Convert mel to Hz
        """
        mo = self.calc_mo()
        return self.fo * (np.exp(m / mo) - 1.0)


    def melfilterbank(self):
        """
        Calculate mel-fileterbank.

        Returns:
            filterbank: (numChannels, N) matrix. 
            fcenters: Frequency of bandpass filtering.
        """
        fmax = self.fs / 2 #Nyquist frequency
        melmax = self.hz2mel(fmax) #Mel scale of Nyquist frequency
        Nmax = int(self.N / 2) #Maximum frequency index
        df = self.fs / self.N #Frequency resolution
        dmel = melmax / (self.numChannels + 1)
        melcenters = np.arange(1, self.numChannels + 1) * dmel #Center frequency of each filter on the Mel scale
        fcenters = self.mel2hz(melcenters) #Converts the center frequency of each filter ot Hz
        indexcenter = np.round(fcenters / df) #Convert the center frequency of each filter to freuency indexes
        indexstart = np.hstack(([0], indexcenter[0:self.numChannels - 1]))
        indexstop = np.hstack((indexcenter[1:self.numChannels], [Nmax]))
        filterbank = np.zeros((self.numChannels, Nmax))

        for channel in np.arange(0, self.numChannels):
            #Find the point from the slope of the straight line to the left of the triangular fileter
            increment = 1.0 / (indexcenter[channel] - indexstart[channel])
            for i in np.arange(indexstart[channel], indexcenter[channel]):
                filterbank[int(channel), int(i)] = (i - indexstart[channel]) * increment
            #Find the point from the slope of the straight line to the right of the triangular filete
            decrement = 1.0 / (indexstop[channel] - indexcenter[channel])
            for j in np.arange(indexcenter[channel], indexstop[channel]):
                filterbank[int(channel), int(j)] = 1.0 - ((j - indexcenter[channel]) * decrement)
        return filterbank, fcenters


if __name__ == "__main__":
    """sample"""
    import soundfile
    fname = 'recordings/0_jackson_0.wav'
    data, fs = soundfile.read(fname)
    N = len(data)
    numChannels = 20
    df = fs / N
    melfilterbank_cal = MelFilterBankCalc(N, fs, numChannels)
    filterbank, fcenters = melfilterbank_cal.melfilterbank()


    """plot graph"""
    import matplotlib.pyplot as plt
    for channel in np.arange(0, numChannels):
        plt.plot(np.arange(0, int(N/2)) * df, filterbank[channel])
    plt.xlabel("frequency [Hz]")
    png_fname = "mel-filterbank_fo=700Hz.png"
    plt.savefig(png_fname)
    plt.show()