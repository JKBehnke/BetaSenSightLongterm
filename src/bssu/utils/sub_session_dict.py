"""" Dictionary of all included subjects with their sessions  """

sub_session_dict = {
    "017": ["fu3m", "fu12m", "fu36m"],
    "019": ["fu3m", "fu12m", "fu18m"],
    "021": ["fu3m", "fu12m", "fu18m"],
    "024": ["postop", "fu3m", "fu12m", "fu18m"],
    "025": ["postop", "fu3m", "fu12m"],
    "026": ["postop", "fu3m", "fu12m", "fu36m"],
    "028": ["postop", "fu12m", "fu24m"],
    "029": ["postop", "fu3m", "fu12m", "fu18m"],
    "030": ["postop", "fu3m", "fu12m", "fu24m"],
    "031": ["postop", "fu3m"],
    "032": ["postop", "fu3m"],
    "033": ["fu3m", "fu12m", "fu18m"],
    "036": ["fu12m", "fu18m"],
    "038": ["postop", "fu3m"],
    "040": ["fu3m", "fu12m", "fu24m"],
    "041": ["fu3m", "fu12m", "fu18m"],
    "045": ["fu3m", "fu12m"],
    "047": ["postop", "fu12m", "fu18m"],
    "048": ["postop", "fu12m", "fu18m"],
    "049": ["postop", "fu12m"],
    "050": ["fu3m", "fu12m", "fu18m"],
    "052": ["postop", "fu12m", "fu18m"],
    "055": ["postop", "fu12m", "fu18m"],
    "059": ["postop", "fu3m", "fu12m", "fu24m"],
    "060": ["postop", "fu3m", "fu24m"],
    "061": ["postop", "fu3m", "fu12m"],
    "062": ["postop", "fu3m", "fu12m"],
    "063": ["postop", "fu3m", "fu12m"],
    "065": ["postop", "fu3m"],
    "066": ["postop", "fu3m", "fu12m"],
    "069": ["postop", "fu3m", "fu12m"],
    "072": ["postop", "fu3m", "fu12m"],
    "075": ["postop", "fu3m", "fu12m"],
    "081": ["postop", "fu3m", "fu12m"],
    "084": ["postop", "fu3m", "fu12m"],
}  # n=34

# added after MDS revision 1:
# "017": "fu36m"
# "026": "fu36m"
# "040": "fu24m"
# "059": "fu24m"
# "060": "fu24m"
# "066": "fu12m"
# "069": ["postop", "fu3m", "fu12m"]
# "072": ["postop", "fu3m", "fu12m"]
# "075": ["postop", "fu3m", "fu12m"]
# "081": ["postop", "fu3m", "fu12m"]
# "084": ["postop", "fu3m", "fu12m"]

included_sub_sessions = {
    "017": ["fu3m", "fu12m", "fu18or24m"],
    "019": ["fu3m", "fu12m", "fu18or24m"],
    "021": ["fu3m", "fu12m", "fu18or24m"],
    "024": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "025": ["postop", "fu3m", "fu12m"],
    "026": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "028": ["postop", "fu12m", "fu24m"],
    "029": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "030": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "031": ["postop", "fu3m"],
    "032": ["postop", "fu3m"],
    "033": ["fu3m", "fu12m", "fu18or24m"],
    # "036": ["fu12m", "fu18m"],
    "038": ["postop", "fu3m"],
    "040": ["fu3m", "fu12m", "fu18or24m"],
    "041": ["fu3m", "fu12m", "fu18or24m"],
    # "045": ["fu3m", "fu12m"],
    "047": ["postop", "fu12m", "fu18m"],
    "048": ["postop", "fu12m", "fu18m"],
    "049": ["postop", "fu12m"],
    "050": ["fu3m", "fu12m", "fu18or24m"],
    "052": ["postop", "fu12m", "fu18m"],
    "055": ["postop", "fu12m", "fu18m"],
    "059": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "060": ["postop", "fu3m", "fu18or24m"],
    "061": ["postop", "fu3m", "fu12m"],
    "062": ["postop", "fu3m", "fu12m"],
    "063": ["postop", "fu3m", "fu12m"],
    "065": ["postop", "fu3m"],
    "066": ["postop", "fu3m", "fu12m"],
    "069": ["postop", "fu3m", "fu12m"],
    "072": ["postop", "fu3m", "fu12m"],
    "075": ["postop", "fu3m", "fu12m"],
    "081": ["postop", "fu3m", "fu12m"],
    "084": ["postop", "fu3m", "fu12m"],
}  # n=33

sub_session_perceive_error = {
    "030": ["fu24m"],
    "055": ["fu18m"],
    "062": ["fu12m"],
    "033": ["fu3m", "fu12m", "fu18m"],
}

sub_session_group_0 = {
    "024": ["postop", "fu3m"],
    "025": ["postop", "fu3m"],
    "026": ["postop", "fu3m"],
    "028": ["postop", "fu12m"],
    "029": ["postop", "fu3m"],
    "030": ["postop", "fu3m"],
    "031": ["postop", "fu3m"],
    "032": ["postop", "fu3m"],
    "038": ["postop", "fu3m"],
    "047": ["postop", "fu12m"],
    "048": ["postop", "fu12m"],
    "049": ["postop", "fu12m"],
    "052": ["postop", "fu12m"],
    "055": ["postop", "fu12m"],
    "059": ["postop", "fu3m"],
    "060": ["postop", "fu3m"],
    "061": ["postop", "fu3m"],
    "062": ["postop", "fu3m"],
    "063": ["postop", "fu3m"],
    "065": ["postop", "fu3m"],
    "066": ["postop", "fu3m"],
    "069": ["postop", "fu3m"],
    "072": ["postop", "fu3m"],
    "075": ["postop", "fu3m"],
    "081": ["postop", "fu3m"],
    "084": ["postop", "fu3m"],
}  # n=26


sub_session_group_1 = {
    "024": ["postop", "fu3m", "fu12m"],
    "025": ["postop", "fu3m", "fu12m"],
    "026": ["postop", "fu3m", "fu12m"],
    "029": ["postop", "fu3m", "fu12m"],
    "030": ["postop", "fu3m", "fu12m"],
    "059": ["postop", "fu3m", "fu12m"],
    "061": ["postop", "fu3m", "fu12m"],
    "062": ["postop", "fu3m", "fu12m"],
    "063": ["postop", "fu3m", "fu12m"],
    "066": ["postop", "fu3m", "fu12m"],
    "069": ["postop", "fu3m", "fu12m"],
    "072": ["postop", "fu3m", "fu12m"],
    "075": ["postop", "fu3m", "fu12m"],
    "081": ["postop", "fu3m", "fu12m"],
    "084": ["postop", "fu3m", "fu12m"],
}  # n=15

sub_session_group_2 = {
    "017": ["fu3m", "fu12m", "fu18or24m"],
    "019": ["fu3m", "fu12m", "fu18or24m"],
    "021": ["fu3m", "fu12m", "fu18or24m"],
    "024": ["fu3m", "fu12m", "fu18or24m"],
    "026": ["fu3m", "fu12m", "fu18or24m"],
    "029": ["fu3m", "fu12m", "fu18or24m"],
    "030": ["fu3m", "fu12m", "fu18or24m"],
    "033": ["fu3m", "fu12m", "fu18or24m"],
    "040": ["fu3m", "fu12m", "fu18or24m"],
    "041": ["fu3m", "fu12m", "fu18or24m"],
    "050": ["fu3m", "fu12m", "fu18or24m"],
    "059": ["fu3m", "fu12m", "fu18or24m"],
}  # n=12

sub_session_group_3 = {
    "024": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "026": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "029": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "030": ["postop", "fu3m", "fu12m", "fu18or24m"],
    "059": ["postop", "fu3m", "fu12m", "fu18or24m"],
}  # n=5


def get_sessions(sub: str):
    """Get sessions for a subject"""
    return sub_session_dict[sub]


def check_if_perceive_error(sub: str):
    """Check if subject has perceive error"""
    if sub in sub_session_perceive_error:
        return sub_session_perceive_error[sub]
    else:
        return "No"


def get_subs_sessions(cohort: str):
    """Get subjects and their sessions for a cohort"""
    if cohort == "all_included":
        sub_ses_dict = included_sub_sessions
    elif cohort == "group_0":
        sub_ses_dict = sub_session_group_0
    elif cohort == "group_1":
        sub_ses_dict = sub_session_group_1
    elif cohort == "group_2":
        sub_ses_dict = sub_session_group_2
    elif cohort == "group_3":
        sub_ses_dict = sub_session_group_3
    else:
        raise ValueError("Invalid cohort")

    return {"incl_subjects": sub_ses_dict.keys(), "incl_sessions": sub_ses_dict}
