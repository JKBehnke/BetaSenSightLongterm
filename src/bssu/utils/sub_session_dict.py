"""" Dictionary of all included subjects with their sessions  """

sub_session_dict = {
    "017": ["fu3m", "fu12m"],
    "019": ["fu3m", "fu12m", "fu18m"],
    "021": ["fu3m", "fu12m", "fu18m"],
    "024": ["postop", "fu3m", "fu12m", "fu18m"],
    "025": ["postop", "fu3m", "fu12m"],
    "026": ["postop", "fu3m", "fu12m"],
    "028": ["postop", "fu12m", "fu24m"],
    "029": ["postop", "fu3m", "fu12m", "fu18m"],
    "030": ["postop", "fu3m", "fu12m", "fu24m"],
    "031": ["postop", "fu3m"],
    "032": ["postop", "fu3m"],
    "033": ["fu3m", "fu12m", "fu18m"],
    "036": ["fu12m", "fu18m"],
    "038": ["postop", "fu3m"],
    "040": ["fu3m", "fu12m"],
    "041": ["fu3m", "fu12m", "fu18m"],
    "045": ["fu3m", "fu12m"],
    "047": ["postop", "fu12m", "fu18m"],
    "048": ["postop", "fu12m", "fu18m"],
    "049": ["postop", "fu12m"],
    "050": ["fu3m", "fu12m", "fu18m"],
    "052": ["postop", "fu12m", "fu18m"],
    "055": ["postop", "fu12m", "fu18m"],
    "059": ["postop", "fu3m", "fu12m"],
    "060": ["postop", "fu3m"],
    "061": ["postop", "fu3m", "fu12m"],
    "062": ["postop", "fu3m", "fu12m"],
    "063": ["postop", "fu3m", "fu12m"],
    "065": ["postop", "fu3m"],
    "066": ["postop", "fu3m"],
}

sub_session_perceive_error = {
    "030": ["fu24m"],
    "055": ["fu18m"],
    "062": ["fu12m"],
    "033": ["fu3m", "fu12m", "fu18m"],
    }


def get_sessions(sub: str):
    """ Get sessions for a subject """
    return sub_session_dict[sub]

def check_if_perceive_error(sub:str):
    """ Check if subject has perceive error """
    if sub in sub_session_perceive_error:
        return sub_session_perceive_error[sub]
    else:
        return "No"