""" feature class """

import pandas as pd
from dataclasses import dataclass


@dataclass (init=True, repr=True)
class featureClass:
    """
    feature Class 
    
    parameters:
        - sub: e.g. "021"
        - feature: str e.g.  
            "PowerSpectrum": 
                ["frequency", "time_sectors", "averagedPSD", "SEM_rawPsd", 
                "normPsdToTotalSum", "SEM_normPsdToTotalSum", "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]

            "PSDaverageFrequencyBands": 
                ["averagedrawPSD"] 
            
            "PeakParameters":
                ["PEAK_frequency", "highest_peak_height_5HzAverage"]

        - metaClass: all original attributes set in Main_Class
        - resultValue: selected meta_table set in modality_class

    Returns:
        - sel_Result_DF: session selected meta_table 
    
    """
    
    sub: str             # note that : is used, not =  
    feature: str
    metaClass: any
    resultValue: any


    def __post_init__(self,):        
        

        print("FEATURECLASS:", self.resultValue)
        # set the attribute for the single value from the Dataframe , so self.data will output for example the array of one single Power Spectrum
        setattr(
            self,
            'data',
            self.resultValue
            )


