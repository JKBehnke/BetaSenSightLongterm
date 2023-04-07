""" channel class """

import pandas as pd
from dataclasses import dataclass

from .. classes import featureAnalysis_class as feature_class
from .. classes import normalizationAnalysis_class as normalization_class


@dataclass (init=True, repr=True)
class channelClass:
    """
    channel Class 
    
    parameters:
        - sub: e.g. "021"
        - channel: str e.g. '03', '13', '02', '12', '01', '23', 
                            '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                            '1A2A', '1B2B', '1C2C' set in session_class
        - metaClass: all original attributes set in main_Class
        - Result_DF: selected meta_table set in session_class

    Returns:
        - sel_Result_DF: channel selected meta_table 
    
    """
    
    sub: str             # note that : is used, not =  
    channel: str
    metaClass: any
    Result_DF: pd.DataFrame


    def __post_init__(self,):        
        

        if self.metaClass.result == "PowerSpectrum":
            ############### FEATURE SELECTION ###############

            allowed_features = ["frequency", "time_sectors", 
                                "rawPsd", "SEM_rawPsd", 
                                "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                                "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                                "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]
            
            
            # go directly to feature class and set the attribute of result Value from the correct feature column
            for feat in self.metaClass.feature:

                # Error checking: if feature is not in allowed_features -> Error message
                assert feat.lower() in [f.lower() for f in allowed_features], (
                    f'inserted feature ({feat}) should'
                    f' be in {allowed_features}'
                )
                
                # there is only one row left in the result dataframe, now get the value from the feature column 
                sel_feature = self.Result_DF[feat].iloc[0] 
                

                if len(sel_feature) == 0:
                    continue
                    
                # set the channel value for each channel in channelClass 
                setattr(
                    self,
                    feat,
                    feature_class.featureClass(
                        sub=self.sub,
                        feature = feat,
                        metaClass=self.metaClass,
                        resultValue=sel_feature
                    ),
                )  

        
        # for results "PSDaverageFrequencyBands" and "PeakParameters": first go to frequency Band Class and then extract feature! 
        elif self.metaClass.result == "PSDaverageFrequencyBands" or "PeakParameters":

            ############### NORMALIZATION SELECTION ###############
            allowed_normalization = ["rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"]

            # Error checking: if normalization is not in allowed_features -> Error message
            for norm in self.metaClass.normalization:

                assert norm in [n for n in allowed_normalization], (
                    f'inserted normalization ({norm}) should'
                    f' be in {allowed_normalization}'
                )            

                # select the correct normalization rows first
                sel_Result_DF = self.Result_DF[self.Result_DF.absoluteOrRelativePSD == norm]
                

                if len(sel_Result_DF) == 0:
                    continue
                    
                # set the normalization value for each normalization in normalizationClass 
                setattr(
                    self,
                    norm,
                    normalization_class.normalizationClass(
                        sub=self.sub,
                        normalization = norm,
                        metaClass=self.metaClass,
                        Result_DF=sel_Result_DF)
                )  

        else: 
            print("result is not defined")



    