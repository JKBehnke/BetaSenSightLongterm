""" session class """

import pandas as pd
from dataclasses import dataclass

import analysis.classes.channelAnalysis_class as channel_class


@dataclass (init=True, repr=True)
class sessionClass:
    """
    session Class 
    
    parameters:
        - sub: e.g. "021"
        - session: str e.g. "postop", "fu3m", "fu12m", "fu18m", "fu24m" set in main_class
        - metaClass: all original attributes set in Main_Class
        - resultDataFrame: selected meta_table set in modality_class

    Returns:
        - sel_Result_DF: session selected meta_table 
    
    """
    
    sub: str             # note that : is used, not =  
    session: str
    metaClass: any
    Result_DF: pd.DataFrame


    def __post_init__(self,):        
        
        allowed_channels = ['03', '13', '02', '12', '01', '23', 
                            '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                            '1A2A', '1B2B', '1C2C']

        # continue to next class: Channel_Class and set the attribute of the new selection of metaClass
        for chan in self.metaClass.pickChannels:

            assert chan in allowed_channels, (
                f'inserted condition ({chan}) should'
                f' be in {allowed_channels}'
            )

            # only get the rows of the meta_table that include the correct conditions in column "condition"
            # sel = [cond.lower() == c.lower() for c in self.meta_table["condition"]]
            # # sel = [cond.lower() == c for c in self.meta_table["condition"]]  
            # sel_meta_table = self.meta_table[sel].reset_index(drop=True) # reset index of the new meta_table 
            
            sel_Result_DF = self.Result_DF[self.Result_DF.bipolarChannel.str.contains(chan)]


            # if no files are left after selecting, dont make new class
            if len(sel_Result_DF) == 0:
                continue

            # values starting with integers can not be set as an attribute, therefore transform to string starting with BIP_
            bipolar_chan = f"BIP_{chan}"

            # set the channel value for each channel in channelClass 
            setattr(
                self,
                bipolar_chan,
                channel_class.channelClass(
                    sub=self.sub,
                    channel = bipolar_chan,
                    metaClass=self.metaClass,
                    Result_DF=sel_Result_DF
                ),
            )  


