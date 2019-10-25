#!/opt/conda/envs/dsenv/bin/python

def filter_cond(line_dict):
    """Filter function
    Takes a dict with field names and values as the argument
    Returns True if conditions are satisfied
    """
    cond_match = False
    if line_dict['if1'].isdigit():
        cond_match = (
           int(line_dict["if1"]) > 20 and int(line_dict["if1"]) < 40
        )
    return True if cond_match else False

