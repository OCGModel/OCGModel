"""
Concatenates definitions that are shared by the different modules and external configurations

Attributes:
    cplex (cplex or False): If the cplex API is found by the python, than cplex is imported else this a
                            attribute is set as False.

    sankey_opacity (float): Opacity of links in the sankey diagram plot.
    Y: "YEAR"
    T: "TECHNOLOGY"
    L: "TIMESLICE"
    F: "FUEL"
    E: "EMISSION"
    M: "MODE_OF_OPERATION"
    R: "REGION"
    LS: "SEASON"
    LD: "DAYTYPE"
    LH: "DAILYTIMEBRACKET"
    S:  "STORAGE"

"""

import configparser  # Read config file
import warnings
import os

# try to import cplex
try:
    import cplex as cplex
except:
    warnings.warn("CPLEX API not found. Using CPLEX as solver will not be possible."
                  "Assure to set GLPK as solver at the settings.ini file.")
    cplex = False

# Global Vars
SETTINGS_FILENAME = "settings.ini"
VERSIONS_WITH_PRODUCTION_SHARES = ["osemosys_short_OCG.txt"]
VERSIONS_WITH_ACTIVITY_PROFILE = ["osemosys_short_OCG.txt"]
VERSIONS_WITH_INFINITY_STORAGE = ["osemosys_short_OCG.txt"]
VERSIONS_WITH_YEARMULTIPLIER = []

# SETS Indexing
Y = "YEAR"
T = "TECHNOLOGY"
L = "TIMESLICE"
F = "FUEL"
E = "EMISSION"
M = "MODE_OF_OPERATION"
R = "REGION"
LS = "SEASON"
LD = "DAYTYPE"
LH = "DAILYTIMEBRACKET"
S = "STORAGE"


# Settings Class
class SettingsException(Exception):
    # Class for throwing Exceptions
    pass


class Settings:
    """Settings object

    Read model settings.ini file and summarize important information to other modules.

    Args:
        settings_file (str): .ini file path

    """

    def __init__(self, settings_file):

        # Path to settings
        self.settings_file = settings_file

        # Settings filename
        self.filename = settings_file.split(os.sep)[-1]

        # Start setting names
        self.comments = dict()

        self.data_dir = None
        self.runs_dir = None
        self.osemosys_code_version = None
        self.osemosys_dir = None
        self.runs_dir = None
        self.solver = None

        self.cplex_input_filename = None
        self.cplex_output_filename = None
        self.cplex_solutions_dir = None
        self.barepcomp = None

        self.glpk_solutions_dir = None

        self.compute_missing_variables = None
        self.sankey_opacity = None

        # Read configs
        self._parser = configparser.ConfigParser()
        self._parser.read(settings_file)

        # read all configurations
        for s in self._parser.sections():
            if s == "Comments":
                for opt in self._parser.options(section=s):
                    self.comments[opt] = self._parser.get(s, opt)
            else:
                for opt in self._parser.options(section=s):
                    setattr(self, opt, self._parser.get(s, opt))

    def update(self):
        """
        Update settings file
        """
        for s in self._parser.sections():
            if s != "Comments":
                for opt in self._parser.options(section=s):
                    if opt in self.__dict__.keys():
                        self._parser.set(s, opt, getattr(self, opt))

        with open(self.settings_file, "w") as configfile:
            self._parser.write(configfile)


# Initialize settings
settings = Settings(SETTINGS_FILENAME)
