""" normalization Class """


import pandas as pd
from dataclasses import dataclass

from .. classes import frequencyBand_class as freqBand_class


@dataclass (init=True, repr=True)
class normalizationClass:
    """
    normalization Class only relevant for results "PSDaverageFrequencyBands" or "PeakParameters"
    
    parameters:
        - sub: e.g. "021"
        - normalization: str e.g. "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz" set in main_class
        - metaClass: all original attributes set in Main_Class
        - resultDataFrame: selected meta_table set in modality_class

    Returns:
        - sel_Result_DF: session selected meta_table 
    
    """
    
    sub: str             # note that : is used, not =  
    normalization: str
    metaClass: any
    Result_DF: pd.DataFrame


    def __post_init__(self,):        
        

        ############### FREQUENCY BAND SELECTION ###############
        allowed_freqBands = ["beta", "lowBeta", "highBeta", "alpha", "narrowGamma"]

        # continue to next class: feature Class and set the attribute of the new selection of metaClass
        for freq in self.metaClass.freqBands:

            # Error checking: if feature is not in allowed_features -> Error message
            assert freq.lower() in [fq.lower() for fq in allowed_freqBands], (
                f'inserted frequency Band ({freq}) should'
                f' be in {allowed_freqBands}'
            )
            
            # select from the normalization filtered Dataframe the correct frequency band
            sel_Result_DF = self.Result_DF[self.Result_DF.frequencyBand == freq]
        

            if len(sel_Result_DF) == 0:
                continue
                
            # set the channel value for each channel in channelClass 
            setattr(
                self,
                freq,
                freqBand_class.freqBandClass(
                    sub=self.sub,
                    freqBand = freq,
                    metaClass=self.metaClass,
                    Result_DF=sel_Result_DF
                ),
            )  

    

