""" frequency Band Class """


import pandas as pd
from dataclasses import dataclass

import analysis.classes.featureAnalysis_class as feature_class


@dataclass (init=True, repr=True)
class freqBandClass:
    """
    frequency Band Class 
    
    parameters:
        - sub: e.g. "021"
        - freqBand: str e.g. "beta", "lowBeta", "highBeta", "alpha", "narrowGamma" set in main_class
        - metaClass: all original attributes set in Main_Class
        - resultDataFrame: selected meta_table set in modality_class

    Returns:
        - sel_Result_DF: session selected meta_table 
    
    """
    
    sub: str             # note that : is used, not =  
    freqBand: str
    metaClass: any
    Result_DF: pd.DataFrame


    def __post_init__(self,):        
        

        if self.metaClass.result == "PSDaverageFrequencyBands":

            allowed_features =  ["averagedPSD"] 
                
                
            # continue to next class: feature Class and set the attribute of the new selection of metaClass
            for feat in self.metaClass.feature:

                # Error checking: if feature is not in allowed_features -> Error message
                assert feat.lower() in [f.lower() for f in allowed_features], (
                    f'inserted contact ({feat}) should'
                    f' be in {allowed_features}'
                )
                
                # there is only one row left in the result dataframe, now get the value from the feature column 
                sel_feature = self.Result_DF[feat].iloc[0] 
        

                # if len(sel_feature) == 0:
                #     continue
                    
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


        elif self.metaClass.result == "PeakParameters":

            allowed_features =  ["PEAK_frequency", "PEAK_amplitude", "PEAK_5HzAverage"] 
                
                
            # continue to next class: feature Class and set the attribute of the new selection of metaClass
            for feat in self.metaClass.feature:

                # Error checking: if feature is not in allowed_features -> Error message
                assert feat.lower() in [f.lower() for f in allowed_features], (
                    f'inserted contact ({feat}) should'
                    f' be in {allowed_features}'
                )
                
                # there is only one row left in the result dataframe, now get the value from the feature column 
                sel_feature = self.Result_DF[feat].iloc[0] 
        

                # if len(sel_feature) == 0:
                #     continue
                    
                # set the value for each feature in featureClass 
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
        
        else: 
            print("result is not defined")


