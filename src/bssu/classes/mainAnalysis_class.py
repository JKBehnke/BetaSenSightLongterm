""" Analysis main class """


# import packages
import os 
from dataclasses import dataclass, field
import pandas as pd
from numpy import array

from .. utils import find_folders as find_folders
from .. utils import loadResults as loadResults
from .. classes import metadataAnalysis_class as metadata
from .. classes import sessionAnalysis_class as session_class



@dataclass(init=True, repr=True) 
class MainClass:
    """
    Main analysis class to extract results from the saved json files with Power Spectra, PSD averages and Peak results

    1) There are 3 different json files from which you can extract data: 
        - filepath: c:\\Users\\jebe12\\Research\\Longterm_beta_project\\Code\\BetaSenSightLongterm\\results\\sub-0XX
        - filename for result of PowerSpectrum: "SPECTROGRAMPSD_{hemisphere}_{filter}.json"
        - filename for result of PSDaverageFrequencyBands: "SPECTROGRAMpsdAverageFrequencyBands_{hemisphere}_{filter}.json"
        - filename for result of PeakParameters: "SPECTROGRAM_highestPEAK_FrequencyBands_{hemisphere}_{filter}.json"
    
    2) depending on input of sub, hemispere, filter and result:
        - one json file is being loaded
        - the json file will be transformed to a Dataframe and saved in its original format in metadata class
        - for results PSDaverageFrequencyBands or PeakParameters: filter the Dataframe already for correct normalization before storing in metadata
        

    3) main_class selects the rows for each session and sets the attribute for session class
    4) session_class selects the rows for each channel and sets the attribute for channel class
    5) channel_class:
        if result = "PowerSpectrum"
            - selects directly the value of each feature and sets the attribute for feature_class
        
        if result = "PSDaverageFrequencyBands" or "PeakParameters"
            - first selects rows for each frequency band and sets the attribute for freqBand_class
    
    6) frequencyBand_class only relevant if result = "PSDaverageFrequencyBands" or "PeakParameters"
        selects the value for each feature sets the attribute for feature_class
    
    7) feature_class:
        sets an attribute "data" to itself containing the value = output

        
    
    parameters:
 
        - incl_sub: str e.g. "024"
        - incl_session: list ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]
        - pickChannels: list of bipolar channels, depending on which incl_contact was chosen
                        Ring: ['03', '13', '02', '12', '01', '23']
                        SegmIntra: ['1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C']
                        SegmInter: ['1A2A', '1B2B', '1C2C']
        - hemisphere: str e.g. "Right"
        - normalization: str "rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"
        - filter: str "unfiltered", "band-pass"
        - result: str "PowerSpectrum", "PSDaverageFrequencyBands", "PeakParameters"

        - feature: list of features you want to extract from the json file, depending on chosen result
            "PowerSpectrum": 
                ["frequency", "time_sectors", 
                "rawPsd", "SEM_rawPsd", 
                "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz"]

            "PSDaverageFrequencyBands": 
                ["frequencyBand", "averagedPSD"] 
            
            "PeakParameters":
                ["PEAK_frequency", "PEAK_amplitude", "PEAK_5HzAverage"]

    TODO: 
        - fix .json files for PSDaverageFrequencyBands and PeakParameters: should contain all normalization variants! 
        - then take out normalization in main_class and metadata_class
        - add normalization features in frequency band class
        - make sure column names are useful: instead of averagedPSD -> rawPsd
    
    Returns:
     
        after running the class and saving it in a variable, 
            e.g. sub-029 = mainAnalysis_class.MainClass(
                        sub="029",
                        hemisphere = "Right",
                        filter = "band-pass",
                        result = "PSDaverageFrequencyBands",
                        incl_session = ["postop", "fu3m", "fu12m", "fu18m"],
                        pickChannels = ['03', '13', '02', '12', '01', '23', 
                                        '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                        '1A2A', '1B2B', '1C2C'],
                        normalization = ["rawPsd", "normPsdToSum1_100Hz"],
                        freqBands = ["beta", "highBeta", "lowBeta"],
                        feature= ["averagedPSD"]
                        )

        call one single feature by this codeline structure,
        
        - result="PowerSpectrum": 
            sub029.postop.BIP_03.rawPsd.data

        - result="PSDaverageFrequencyBands": 
            sub-029.postop.BIP_03.rawPsd.highBeta.averagedPSD.data

        - result="PeakParameters": 
            sub-029.postop.BIP_03.rawPsd.beta.PEAK_5HzAverage.data
    """

    # these fields will be initialized 
    sub: str             # note that : is used, not =  
    hemisphere: str 
    filter: str
    result: str
    incl_session: list = field(default_factory=lambda: ["postop", "fu3m", "fu12m", "fu18m", "fu24m"]) # default:_ if no input is given -> automatically input the full list
    pickChannels: list = field(default_factory=lambda: ['03', '13', '02', '12', '01', '23', 
                                                        '1A1B', '1B1C', '1A1C', '2A2B', '2B2C', '2A2C', 
                                                        '1A2A', '1B2B', '1C2C'])
    normalization: list = field(default_factory=lambda: ["rawPsd", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"])
    freqBands: list = field(default_factory=lambda: ["beta", "lowBeta", "highBeta", "alpha", "narrowGamma"])
    feature: list = field(default_factory=lambda: ["frequency", "time_sectors", "rawPsd", "SEM_rawPsd", 
                                                   "normPsdToTotalSum", "SEM_normPsdToTotalSum", 
                                                   "normPsdToSumPsd1to100Hz", "SEM_normPsdToSumPsd1to100Hz",
                                                   "normPsdToSum40to90Hz", "SEM_normPsdToSum40to90Hz",
                                                   "frequencyBand", "averagedPSD", 
                                                   "PEAK_frequency", "PEAK_amplitude", "PEAK_5HzAverage", 
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




