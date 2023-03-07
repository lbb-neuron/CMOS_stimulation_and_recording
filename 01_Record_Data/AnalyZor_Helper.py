import numpy as np

def constant_getter(constant):
    """
    This is a function in which all MaxOne System constants are stored. Correct keyword will return constant.
    """
    if constant == "chipWidth":
        # chip has 220 electrodes width
        return 220
    elif constant == "chipHeight":
        # chip is 120 electrodes high
        return 120
    elif constant == "samplingTime":
        # sampling time is 50 microseconds
        return 0.00005
    elif constant == "sampling Rate":
        # sampling rate is 20 kHz
        return 20000
    else:
        raise ValueError("Constant not known or implemented yet.")

def convert_xy_to_elno(x, y):
    """
    Convert coordinates (x,y) to an electrode number from 0 to 26'399.
    This function uses numpy convention (x is axis 0, y is axis 1)
    :param x: (0-119) is row
    :param y: (0-219) is column
    """
    chipWidth = constant_getter("chipWidth")
    
    return x*chipWidth + y%chipWidth

def convert_elno_to_xy(elno):
    """
    Convert electrode number (0-26399) to electrode coordinates (x,y).
    This function uses numpy convention (x is axis 0, y is axis 1)
    :param elno: (0-26'399) is the absolute electrode number
    """
    chipWidth = constant_getter("chipWidth")
    x = int(elno/chipWidth)
    y = elno % chipWidth
    return x,y

def generate_mask(electrodes_to_analyze):
    """
    Generates a numpy matrix in which each routed electrode is set to 1
    :param electrodes to analyze: list of electrodes that are part of the microstructure
    """
    cmos_mea = np.empty((120,220))
    cmos_mea[:] = np.nan

    for electrode in electrodes_to_analyze:
        coords = convert_elno_to_xy(electrode)
        cmos_mea[coords[0], coords[1]] = 1
    return cmos_mea

# this function should be in a helper class
def generate_grid(self):
    x, y = np.meshgrid(np.linspace(self.boundY[0], self.boundY[1], self.boundY[1] - self.boundY[0]+1), np.linspace(self.boundX[0], self.boundX[1], self.boundX[1] - self.boundX[0]+1) )
    return x,y

