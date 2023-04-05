""" Metadata Class """

from dataclasses import dataclass, field


@dataclass(init=True, repr=True) 
class MetadataClass:
    """
    Metadata class to store repetitive variables that are changing constantly throughout the hierarchy
    
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
        - normalization: list, e.g. ["rawPSD", "normPsdToTotalSum", "normPsdToSum1_100Hz", "normPsdToSum40_90Hz"]
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
    

    sub: str             # note that : is used, not =  
    hemisphere: str 
    filter: str
    result: str
    incl_session:  list
    pickChannels: list      
    normalization: list 
    freqBands: list  
    feature: list
    original_Result_DF: any