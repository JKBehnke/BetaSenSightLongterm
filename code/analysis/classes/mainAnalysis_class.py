""" Analysis main class """


# import packages
import os 
from dataclasses import dataclass, field
import pandas as pd
from numpy import array

import analysis.utils.find_folders as find_folders
import analysis.utils.loadResults as loadResults
import analysis.classes.metadataAnalysis_class as metadata
import analysis.classes.sessionAnalysis_class as session_class



@dataclass(init=True, repr=True) 
class MainClass:
    """
    Main analysis class to store results
    
    parameters:
 
        - incl_sub: str e.g. "024"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - incl_condition: list e.g. ["m0s0", "m1s0"]
        - incl_contact: a list of contacts to include ["RingR", "SegmIntraR", "SegmInterR", "RingL", "SegmIntraL", "SegmInterL"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - hemisphere: str e.g. "Right"
        - normalization: str "rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - filter: str "unfiltered", "band-pass"
        - result: str "PowerSpectrum", "PSDaverageFrequencyBands", "PeakParameters"

        - feature: list of features you want to extract from the json file, depending on chosen result
            "PowerSpectrum": 
                ["frequency", "time_sectors", "averagedPSD", "SEM_rawPSD", 
                "normPsdToTotalSum", "SEM_normPsdToTotalSum", "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]

            "PSDaverageFrequencyBands": 
                ["frequencyBand", "averagedrawPSD"] 
            
            "PeakParameters":
                ["PEAK_frequency", "highest_peak_height_5HzAverage"]

    post-initialized parameters:
    
    Returns:
        - 
    """

    # these fields will be initialized 
    sub: str             # note that : is used, not =  
    hemisphere: str 
    filter: str
    normalization: str
    result: str
    incl_session: list = field(default_factory=lambda: ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]) # default:_ if no input is given -> automatically input the full list
    pickChannels: list = field(default_factory=lambda: ['03', '13', '02', '12', '01', '23', 
                                                        '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                                        '1A2A', '1B2B', '1C2C'])
    freqBands: list = field(default_factory=lambda: ["beta", "lowBeta", "highBeta", "alpha", "narrowGamma"])
    feature: list = field(default_factory=lambda: ["frequency", "time_sectors", "averagedPSD", "SEM_rawPsd", 
                                                   "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                                                   "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                                                   "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz",
                                                   "frequencyBand", "averagedrawPSD", 
                                                   "PEAK_frequency", "highest_peak_height_5HzAverage", 
                                                   ])
    
    
    

    def __post_init__(self,):

        allowed_results = ["PowerSpectrum", "PSDaverageFrequencyBands", "PeakParameters"]
        allowed_sessions = ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]

        # Check if result is inserted correctly
      
        assert self.result in allowed_results, (
            f'inserted result ({self.result}) should'
            f' be in {allowed_results}'
        )

        # path to results folder of the subject
        self.sub_results_path = find_folders.get_local_path(folder="results", sub=self.sub)


        # load the correct json file
        self.jsonResult = loadResults.load_PSDjson(
            sub = self.sub,
            result = self.result, # self.result has to be a list, because the loading function is 
            normalization = self.normalization,
            hemisphere = self.hemisphere,
            filter = self.filter
        )

        # make a Dataframe from the JSON file to further select
        self.Result_DF = pd.DataFrame(self.jsonResult)

    
        # define and store all variables in self.metaClass, from where they can continuously be called and modified from further subclasses
        self.metaClass = metadata.MetadataClass(
            sub = self.sub,
            hemisphere = self.hemisphere,
            filter = self.filter,
            normalization = self.normalization,
            result = self.result,
            incl_session = self.incl_session,
            pickChannels = self.pickChannels,
            freqBands = self.freqBands,
            feature = self.feature,
            original_Result_DF = self.Result_DF

        )


        # loop through every session input in the incl_session list 
        # and set the session value for each session
        for ses in self.incl_session:

            assert ses in allowed_sessions, (
                f'inserted session ({ses}) should'
                f' be in {allowed_sessions}'
            )

            # only get the rows of the Result_DF that include the correct sessions in column "session"
            # sel = [ses.lower() == s.lower() for s in self.Result_DF["session"]]
            # # sel = [cond.lower() == c for c in self.meta_table["condition"]]  
            # sel_Result_DF = self.Result_DF[sel].reset_index(drop=True) # reset index of the new meta_table 
            
            sel_Result_DF = self.metaClass.original_Result_DF[self.metaClass.original_Result_DF.session == ses]
            print("RESULTMAIN:", sel_Result_DF)

            # if no files are left after selecting, dont make new class
            if len(sel_Result_DF) == 0:
                continue

            # set the session value for each session in SessionClass 
            setattr(
                self,
                ses,
                session_class.sessionClass(
                    sub=self.sub,
                    session = ses,
                    metaClass=self.metaClass,
                    Result_DF=sel_Result_DF
                ),
            )  




