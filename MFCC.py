import numpy as np
import MelFilterBank
import ModificationTool
import scipy.fftpack.realtransforms

class MFCCclass:
    """
    Class for calculating MFCC(Mel-frequency cepstral coefficients)

    Attributes:
        data: time series data.
        fs: samplingrate.
        cutpoint: Number of data arrays to get from the number of channels in the Mel-fileter bank.
    """
    def __init__(self, data, fs, cutpoint=12, numChannels=20):
        self.data = data
        self.fs = fs
        self.cutpoint = cutpoint
        self.numChannels = numChannels


    def MFCC(self):
        """
        You can calculate the MFCC by calling this function.

        Returns:
            output_ceps: Output of cut-off MFCC.
        
        Raise:
            ValueError: cutpoint must be less than or equal to numChannnel.
        """
        if self.cutpoint > self.numChannels:
            raise ValueError("cutpoint must be less than or equal to numChannnel.")
        
        N = len(self.data)
        dft = np.abs(np.fft.fft(self.data))[:int(N/2)]
        melfileterbank_obj = MelFilterBank.MelFilterBankCalc(N, self.fs)
        filterbank, fcenters = melfileterbank_obj.melfilterbank()
        inner_product_fbank = np.dot(dft, filterbank.T)
        modify_dot = ModificationTool.assign_zero2mean(inner_product_fbank)
        mspec = np.log10(modify_dot)
        ceps = scipy.fftpack.realtransforms.dct(mspec, norm="ortho")
        output_ceps = ceps[1:self.cutpoint + 1]
        return output_ceps