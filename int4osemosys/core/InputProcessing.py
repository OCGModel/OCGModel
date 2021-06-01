"""
Tools to construct an Osemosys Model from the excel model map or from an previous input file
"""

import os
import pandas as pd
import numpy as np
import math
from typing import Iterable, Mapping, Dict, List

import xlrd  # read model map
import scipy.interpolate  # Interpolate input parameters

from core.Settings import SETTINGS_FILENAME, VERSIONS_WITH_ACTIVITY_PROFILE, VERSIONS_WITH_PRODUCTION_SHARES, \
    VERSIONS_WITH_YEARMULTIPLIER, VERSIONS_WITH_INFINITY_STORAGE,  Settings

from core.Settings import Y, T, F, E, M, R, L, LS, LD, LH, S


def _myinterpol_(y, years, vals, last_valid_year=3000, first_valid_year=0):
    if y >= last_valid_year:
        return np.nan
    elif y <= first_valid_year:
        return np.nan

    if len(years) > 1:
        f = scipy.interpolate.interp1d(years, vals, bounds_error=False, fill_value=(vals[0], vals[-1]))
    else:
        f = lambda x: vals[0] if x == years[0] else np.nan

    return f(y)


def _get_interpolation_function(values: List):
    """
    Get the interpolation function for multi year parameters

    :param values: parameters to make the interpolate function.  Must be in the form: [YYYY value ; YYYY value; ...]
    :type values: Iterable[Sting
    :return: function to calculate the interpolation of a Year Y.
        If Y inside the defined interval- returns interpolation;
        If Y outside the interval - return the value of the interval limit;
        If one of the boundaries is NaN returns NaN for every year except the defined one
    :rtype: scipy.interpolate.interp1d
    """

    yy = []
    vals = []
    first_valid_year = 0
    last_valid_year = 3000

    points = values.split(";")
    for pair in points:
        pair = pair.split(" ")
        # remove additional white spaces
        [year, val] = [x for x in pair if x != ""]
        year = int(year)
        if val.lower() != "nan":
            yy.append(int(year))
            vals.append(float(val))
        else:
            if yy == []:
                first_valid_year = year
            else:
                last_valid_year = year

    return lambda x: _myinterpol_(x, yy, vals, last_valid_year = last_valid_year, first_valid_year=first_valid_year)
#    if len(yy) > 1:
#        f = scipy.interpolate.interp1d(yy, vals, bounds_error=False, fill_value=(vals[0], vals[-1]))
#    else:
#        f = lambda x: vals[0] if x == yy[0] else np.nan
#    return f


class MapProcessingException(Exception):
    """
    Class for throwing Exceptions during the map processing
    """
    pass


class OsemosysSet:
    """Class for constructing Osemosys sets

    Args:
        name (str): Set Name
        dim (int): Set dimension
        elements (Iterable, optional): Elements to be added to set. Default None.
        numeric (bool, optional): Indicates weather the set is numeric. Default False.
    """
    def __init__(self, name: str, dim: int, elements: Iterable = None, numeric=False):
        self.name = name
        self.dim = dim
        self.numeric = numeric

        if elements is None:
            self.elm = []
        else:
            self.elm = list(elements)  # Set elements

    def __repr__(self):
        return self.elm.__repr__()

    def __getitem__(self, item):
        return self.elm[item]

    def append(self, elem):
        """Add new element to the set.

        Args:
            elem: Element or list of elements to be added to set.
        """
        if isinstance(elem, Iterable):
            if self.numeric:
                self.elm = self.elm + [int(x) for x in elem]
            else:
                self.elm = self.elm + [str(x) for x in elem]
        else:
            if self.numeric:
                self.elm.append(int(elem))
            else:
                self.elm.append(str(elem))

    def write(self, f):
        """
        Write set to OSEMOSYS input file.

        Args:
            f(file): File stream.
        """

        f.write('set %s := \n' % self.name)
        for v in self.elm:
            if self.dim > 1:
                for vv in v:
                    f.write("%s " % vv)
                f.write("\n")
            else:
                f.write('%s \n' % v)
        f.write('; \n\n')


class OsemosysParam:
    """Class for constructing Osemosys Parameters

    Args:
        name (str): Param name
        set_list (List[OsemosysSet]): List with the sets that index this parameter.

        scaling_factor(float, optional): Scaling factor. Default 1.
        default (float, optional): Default value of parameter. Default None.
        group (str, optional): Identifier of which groups this parameter belongs. Default None.
        type (str, optional): Parameter type. Default None.
            Possible types: "Temporal"
        normalize (bool, optional): Indicator weather the values should be normalized to sum 1. Default None.
            Necessary for adjusting the parameter values to the model constraints,  e.g. demand time profiles.

    Attributes:
        set_names(List[str]): List with the same of the sets that index this parameter.

     """

    def __init__(self, name: str, set_list: List[OsemosysSet], scaling_factor: float = 1, default: float = None,
                 group: str = None, type=None, normalize=None):

        self.name = name
        self.set_list = set_list
        self.default = default
        self.group = group
        self.type = type
        self.normalize = normalize

        # if temporal and normalize not given --> default is not normalize!
        if type == "Temporal" and normalize is None:
            self.normalize = False

        self.dim = 0
        self.set_names = []

        for s in set_list:
            self.dim = self.dim + s.dim
            self.set_names.append(s.name)
        self.vals = dict()
        self.scaling_factor = scaling_factor

    def __repr__(self):
        """Represents parameter by defined values"""
        return self.vals.__repr__()

    def __getitem__(self, item: List[str]):
        """
        Get parameter value for a given key.

        Args:
            item(Iterable[str]): List with the parameter key.

        Returns:
            float: Parameter value.
                If given key is valid, but no entry was defined, the method returns the default parameter value.
        Raises:
            MapProcessingException: If key is not valid within parameter sets.
        """
        try:
            item = self._valid_key(item)
            key = self._generate_key(item)
        except MapProcessingException as e:
            raise e

        try:
            ret = self.vals[key]
        except KeyError:  # Key is valid but not defined because param is default
            ret = self.default

        return ret

    def add_value(self, key: List[str], val:float):
        """
        Add value to parameter list

        Args:
            key(Iterable[str]): Parameter index.
            val (float): Parameter Value

        """
        self._valid_key(key)
        nkey = self._generate_key(key)
        self.vals[nkey] = val * self.scaling_factor

    def _generate_key(self, key: List[str]) -> str:
        """
        Convert iterable with index to key string

        Args:
            key(Iterable[str]): Parameter index in list form

        Returns:
            str: Parameter key string.

        """
        key = self._valid_key(key)
        if self.dim > 0:
            nkey = " ".join([str(x) for x in key])
            nkey = str(nkey)
        else:
            nkey = str(key)
        return nkey

    def _valid_key(self, key: List):
        """Verify if key is within parameter index sets and correct numeric keys.

        Args:
            key (List[str]):  Key to be validated

        Returns:
            Iterable: Validated key

        Raises:
            MapProcessingError: The given key is not valid.
                The dimension might be wrong or a parameter is not defined in the sets.

        """
        key_validated = []

        if len(key) != self.dim:
            raise MapProcessingException(
                "%s is invalid key for parameter %s-  Invalid Key Dimension!" % (key, self.name))
        else:
            for i in range(len(self.set_list)):
                ki = key[i]
                if self.set_list[i].numeric:
                    ki = int(float(key[i]))
                else:
                    ki = str(key[i])
                if ki not in self.set_list[i].elm:
                    raise MapProcessingException("%s is invalid key  for parameter %s-  %s not in set %s" % (
                        key, self.name, key[i], self.set_list[i].name))
                key_validated.append(ki)
            return key_validated

    def write(self, f):
        """
        Write parameter to OSEMOSYS input file.

        Args:
            f(file): File stream.
        """

        if self.normalize:
            form_fun = lambda x: "%.15f" % x
        else:
            form_fun = lambda x: "%.10f" % x
        if self.default is not None:
            f.write('param %s default ' % self.name + form_fun(self.default) + ' := \n')
        else:
            f.write('param %s := \n' % self.name)
        if self.dim == 0:
            f.write("%s" % self.vals)
        else:
            for key in self.vals.keys():
                if not math.isnan(self.vals[key]):
                    f.write("%s " % key + form_fun(self.vals[key]) + "\n")
        f.write(";\n\n")


class OsemosysInputGenerator:
    """OSeMOSYS Input Generator

    Initialize Osemosys Model by creating empty sets and parameters.

    Args:
        name(str): Model name
        scenario(str): Scenario name
        data_dir(str): Path to Map file
        sim_dir(str): Path to simulation dir

    Attributes:
        sets(dict[str, OsemosysSet]): Model defined parameters indexed by parameter name.
        params(dict[str,OsemosysParam]): Model defined sets indexed by set name.

    """

    def __init__(self, name: str, scenario: str, data_dir: str, sim_dir: str):
        self.name = name
        self.scenario = scenario

        # If data_dir is not defined, method read from map cannot be perfomed!
        self.data_dir = data_dir
        if data_dir is not None:
            self.read_only = False
            map_name = name + ".xlsx"
            self.map_name = os.sep.join([data_dir, map_name])
            self.ts_folder = os.sep.join([data_dir, "timeseries"])

        self.time_setting = None  # Must be read from Map

        self.sett = None

        # try reading settings file in sim_folder
        try:
            self.sett = Settings(os.sep.join([sim_dir, SETTINGS_FILENAME]))
            self.osemosys_version = self.sett.osemosys_code_version
        except FileNotFoundError:
            self.osemosys_version = None
            pass

        self.dat_folder = sim_dir

        sets_to_build = {1: [Y, T, L, F, E, M, R, LS, LD, LH, S]} # Set names defined in Utils file

        self.sets: Mapping[str, OsemosysSet] = dict()
        self._initialize_sets(sets_to_build)

        # Add params
        self.params = dict()

        # Params to build {paramName: dict(sets = [set, set, set], default = default,)), }

        global_params = {'YearSplit': dict(sets=[L, Y], type="Temporal", normalize=True),
                         'DiscountRate': dict(sets=[R, T], default=0.09),
                         'DaySplit': dict(sets=[LH, Y]),
                         'Conversionls': dict(sets=[L, LS], default=0),
                         'Conversionld': dict(sets=[LD, L], default=0, type="Temporal"),
                         'Conversionlh': dict(sets=[LH, L], default=0, type="Temporal"),
                         'DaysInDayType': dict(sets=[LS, LD, Y], default=0),
                         'TradeRoute': dict(sets=[R, R, F, Y], default=0),
                         'DepreciationMethod': dict(sets=[R], default=1)}
        demands_params = {'SpecifiedAnnualDemand': dict(sets=[R, F, Y], default=0),
                          'SpecifiedDemandProfile': dict(sets=[R, F, L, Y], default=0, type="Temporal", normalize=True),
                          'AccumulatedAnnualDemand': dict(sets=[R, F, Y], default=0)}
        performance_params = {'CapacityToActivityUnit': dict(sets=[R, T], default=8.76),
                              'CapacityFactor': dict(sets=[R, T, L, Y], default=1, type="Temporal"),
                              'AvailabilityFactor': dict(sets=[R, T, Y], default=1),
                              'OperationalLife': dict(sets=[R, T], default=1),
                              'ResidualCapacity': dict(sets=[R, T, Y], default=0),
                              'InputActivityRatio': dict(sets=[R, T, F, M, Y], default=0, type="FUEL_IN"),
                              'OutputActivityRatio': dict(sets=[R, T, F, M, Y], default=0,
                                                          type="FUEL_OUT")}  # Relates technology with fuel output
        technology_costs_params = {'CapitalCost': dict(sets=[R, T, Y], default=0.0001),
                                   'VariableCost': dict(sets=[R, T, M, Y], default=0),
                                   'FixedCost': dict(sets=[R, T, Y], default=0)}
        storage_params = {'TechnologyToStorage': dict(sets=[R, T, S, M], default=0),
                          'TechnologyFromStorage': dict(sets=[R, T, S, M], default=0),
                          'StorageLevelStart': dict(sets=[R, S], default=0),
                          'StorageMaxChargeRate': dict(sets=[R, S], default=99),
                          'StorageMaxDischargeRate': dict(sets=[R, S], default=99),
                          'MinStorageCharge': dict(sets=[R, S, Y], default=0),
                          'OperationalLifeStorage': dict(sets=[R, S], default=0),
                          'CapitalCostStorage': dict(sets=[R, S, Y], default=0),
                          'ResidualStorageCapacity': dict(sets=[R, S, Y], default=0)}
        capacity_constraints_params = {'CapacityOfOneTechnologyUnit': dict(sets=[R, T, Y], default=0),
                                       'TotalAnnualMaxCapacity': dict(sets=[R, T, Y], default=99999),
                                       'TotalAnnualMinCapacity': dict(sets=[R, T, Y], default=0),
                                       'TotalAnnualMaxCapacityInvestment': dict(sets=[R, T, Y], default=999999),
                                       'TotalAnnualMinCapacityInvestment': dict(sets=[R, T, Y], default=0)}
        activity_constraints_params = {'TotalTechnologyAnnualActivityUpperLimit': dict(sets=[R, T, Y], default=999999),
                                       'TotalTechnologyAnnualActivityLowerLimit': dict(sets=[R, T, Y], default=0),
                                       'TotalTechnologyModelPeriodActivityUpperLimit': dict(sets=[R, T],
                                                                                            default=999999),
                                       'TotalTechnologyModelPeriodActivityLowerLimit': dict(sets=[R, T], default=0),
                                       }
        reserve_margin_params = {"ReserveMarginTagTechnology": dict(sets=[R, T, Y], default=0),
                                 "ReserveMarginTagFuel": dict(sets=[R, F, Y], default=0),
                                 "ReserveMargin": dict(sets=[R, Y], default=0)
                                 }
        re_gen_params = {'RETagTechnology': dict(sets=[R, T, Y], default=0),
                         'RETagFuel': dict(sets=[R, F, Y], default=0),
                         'REMinProductionTarget': dict(sets=[R, Y], default=0)}
        emissions_params = {'EmissionActivityRatio': dict(sets=[R, T, E, M, Y], default=0),
                            'EmissionsPenalty': dict(sets=[R, E, Y], default=0),
                            'AnnualExogenousEmission': dict(sets=[R, E, Y], default=0),
                            'AnnualEmissionLimit': dict(sets=[R, E, Y], default=99999),
                            'ModelPeriodExogenousEmission': dict(sets=[R, E], default=0),
                            'ModelPeriodEmissionLimit': dict(sets=[R, E], default=99999)}

        # Parameters not defined for all code versions - Only loaded if the osemosys code version config available:
        if self.osemosys_version is not None:
            # ActivityProfile
            if self.osemosys_version in VERSIONS_WITH_ACTIVITY_PROFILE:
                activity_constraints_params['ActivityProfile'] = dict(sets=[R, T, L, Y], default=-1, type="Temporal",
                                                                      normalize=True)
            # Production shares
            if self.osemosys_version in VERSIONS_WITH_PRODUCTION_SHARES:
                activity_constraints_params['MinAnnualProductionShare'] = dict(sets=[R, T, F, Y], default=-1,
                                                                               type="FUEL_OUT")
                activity_constraints_params['MaxAnnualProductionShare'] = dict(sets=[R, T, F, Y], default=-1,
                                                                               type="FUEL_OUT")
            # Year Multiplier
            if self.osemosys_version in VERSIONS_WITH_YEARMULTIPLIER:
                global_params["YearMultiplier"] = dict(sets=[Y])

            # Infinity Storage
            if self.osemosys_version in VERSIONS_WITH_INFINITY_STORAGE:
                demands_params["InfiniteStorage"] = dict(sets=[F], default=0)

        self._add_params(global_params, group="Global")
        self._add_params(demands_params, group="Demands")
        self._add_params(performance_params, group="Performance")
        self._add_params(technology_costs_params, group="Technology Costs")
        self._add_params(storage_params, group="Storage")
        self._add_params(capacity_constraints_params, group="Capacity Constraints")
        self._add_params(activity_constraints_params, group="Activity Constraints")
        self._add_params(reserve_margin_params, group="Reserve margin")
        self._add_params(re_gen_params, group="RE Generation target")
        self._add_params(emissions_params, group="Emissions")

    @property
    def set_names(self) -> List[str]:
        """Name of all Sets"""
        return list(self.sets.keys())

    @property
    def param_names(self) -> List[str]:
        """Name of all Parameters"""
        return list(self.params.keys())

    @property
    def emissions_params(self) -> Dict[str, OsemosysParam]:
        """All parameters of "Emissions" group"""
        return self._get_params_by_attr(attr_name="group", attr="Emissions")

    @property
    def temporal_params(self) -> Dict[str, OsemosysParam]:
        """All parameters of that are "Temporal" """
        return self._get_params_by_attr(attr_name="type", attr="Temporal")

    def read_model_map(self):
        """Populate the model with the excel model map parameters values.

        Raises:
            MapProcessingException: The excel file with the model parameters could not be found.
                The file must have the same name as the model and be located at the data dir.

        """
        if self.data_dir is None:
            raise MapProcessingException("Object has no data_dir defined. No Map can be read!")
        try:
            mmap = xlrd.open_workbook(self.map_name)
        except FileNotFoundError:
            raise MapProcessingException("Model Map %s not found!" % self.map_name)

        self._load_general_sets(mmap, set_name="Fuel")
        self._load_general_sets(mmap, set_name="Technology")
        self._load_general_sets(mmap, set_name="Emission")
        self._load_scenario_sets(mmap)

        if "YearMultiplier" in self.params.keys():
            self._calculate_year_multipliers()

        self._load_params_in_sheet(mmap, "Emission")
        self._load_params_in_sheet(mmap, "Fuel")
        self._load_params_in_sheet(mmap, "Technology")
        self._load_params_in_sheet(mmap, "Scenario Settings")

        self._load_time_settings(mmap)
        self._load_temporal_params_in_sheet(mmap, "Technology")
        self._load_temporal_params_in_sheet(mmap, "Fuel")

        self._load_plotting_settings()

    def _load_plotting_settings(self):
        """Read plotting settings from Map file and add the config csv file to the simulation dir."""
        df = pd.read_excel(self.map_name, sheet_name="Plotting Settings")
        df = df.dropna()
        df.to_csv(os.sep.join([self.dat_folder, "plotSettings.csv"]))

    def _load_time_settings(self, mmap: xlrd.book.Book):
        """Load Time Settings.

        The time setting is read from the map. The respective time series file is opened in the time series folder.
        With the information in the file the TIMESLICES set and the YearSplit parameters are set accordingly.

        Args:
            mmap (xlrd.book.Book): Model map file streamer.

        Raises
            MapProcessingException: The Time setting is not defined in the Map.

        """

        ws_ts = mmap.sheet_by_name("Time Settings")
        hh_ts = self._get_headers(ws_ts)

        ts_col = hh_ts.index("Time Setting")
        ty_col = hh_ts.index("Type")

        defined_ts = ws_ts.col_values(ts_col)
        if self.time_setting not in defined_ts:
            raise MapProcessingException("Time Setting %s not defined in the Time Settings Sheet!" % self.time_setting)
        else:
            ts_row = ws_ts.col_values(ts_col).index(self.time_setting)
            ts_type = ws_ts.cell_value(ts_row, ty_col)

        # Open and read Time Setting file
        if ts_type == "selection":
            self.time_setting_df = None
            ts = pd.read_csv(os.sep.join([self.ts_folder, self.time_setting]) + ".csv", header=None)
            ts_values = ts.loc[:, 3].values
            self.sets["TIMESLICE"].append(ts_values)
            # How much of the year each TS represents?? 1/total time steps (
            self.params["YearSplit"].default = 1 / len(ts_values)

        elif ts_type == "mean":
            self.time_setting_df = pd.read_csv(os.sep.join([self.ts_folder, self.time_setting]) + ".csv")

            # Define set
            ts_values = self.time_setting_df.loc[:, "timeslice"].values
            self.sets[L].append(ts_values)

            # Define Year split
            self.time_setting_df["w"] = self.time_setting_df["w"].apply(lambda x: x / sum(self.time_setting_df["w"]))
            for ts in self.sets[L]:
                for y in self.sets[Y]:
                    self.params["YearSplit"].add_value(key=[ts, y], val=self.time_setting_df.loc[ts, "w"])

    def _load_temporal_params_in_sheet(self, mmap: xlrd.book.Book, sheet_name):
        """
        Add all the temporal parameters defined in the map sheet to the model.

        Args:
            mmap : Model map file streamer.
            sheet_name (str): Map sheet name. Can be "Technology" or "Fuel".

        Raises:
            MapProcessingError: Sheet does not contain any Temporal parameter.

        """
        # Validate sheet name
        if sheet_name not in ["Technology", "Fuel"]:
            raise MapProcessingException(
                "No temporal parameter to be loaded at Sheet %s! Only Fuels and Technologies are valid" % sheet_name)

        ws = mmap.sheet_by_name(sheet_name)
        hh = self._get_headers(ws, header_index=1)

        cc = self._get_dict_with_col_indexes_for_sets(hh)
        cc["TIMESLICE"] = None  # add to validation!

        temporal_params = self.temporal_params.keys()
        in_sheet_defined_profiles = [x for x in temporal_params if x + "_ProfileName" in hh]

        for p in in_sheet_defined_profiles:
            sets = self.params[p].set_names
            # verify sets -> must be defined
            if any(x not in cc.keys() for x in sets):
                raise MapProcessingException("Parameter is %s is defined in Sheet %s, but index information given not "
                                             "enough" % (p, sheet_name))
            # iterate lines
            param_col = ws.col_values(hh.index(p + "_ProfileName"))
            for r in range(2, ws.nrows):
                profile_name = param_col[r]
                if profile_name != "":  # Not empty
                    if self.time_setting_df is None:
                        with open(os.sep.join([self.ts_folder, profile_name]) + ".txt") as f:
                            ts = f.read().split(" ")
                            ts = [float(x) for x in ts if x != ""]
                            ts.insert(0, 0)  # add to match index
                            ts = np.array(ts)  # Array to allow indexing with timeslices!
                            ts = ts[self.sets["TIMESLICE"].elm]

                            # Potential Normalization (for demand profiles)
                            if self.params[p].normalize:
                                ts = ts / sum(ts)
                    else:
                        try:
                            ts = self.time_setting_df.loc[:, profile_name + ".txt"].values
                            if self.params[p].normalize:
                                ts = np.multiply(ts, self.time_setting_df["w"])
                                ts = ts / sum(ts)
                        except IndexError:
                            raise MapProcessingException(
                                "Time series %s not in time settings .csv file %s" % (profile_name, self.time_setting))

                    for y in self.sets["YEAR"].elm:
                        for i in range(len(self.sets["TIMESLICE"].elm)):
                            l = self.sets["TIMESLICE"][i]
                            help_fun = lambda x: y if x == "YEAR" else l
                            key = [ws.cell_value(r, cc[x]) if x not in ["YEAR", "TIMESLICE"] else help_fun(x) for x
                                   in sets]
                            self.params[p].add_value(key, ts[i])

    def _initialize_sets(self, sets_to_build):
        """Create empty sets with the given names and add to the model.

        Args:
            sets_to_build (dict[int: list]): Sets to be added to the model.
                Dictionary where the keys are the dimension of the sets and the elements a list with the sets names.
        """
        for dim in sets_to_build:
            for s in sets_to_build[dim]:
                if s in [M, L, Y]:
                    self.sets[s] = OsemosysSet(s, dim, numeric=True)
                else:
                    self.sets[s] = OsemosysSet(s, dim)

    def _add_params(self, param_dict, group: str = None):
        """Add parameter to the model.

        Args:
            param_dict(dict): Mapping with the parameter information needed for the OsemosysParam constructor.
            group(str, optional): Parameter group name

        """
        for p in param_dict.keys():
            kargs = param_dict[p]
            set_names = kargs.pop("sets")
            set_list = []
            if len(set_names) != 0:  # dim != 0
                for s in set_names:
                    set_list.append(self.sets[s])
            kargs["group"] = group
            self.params[p] = OsemosysParam(p, set_list, **kargs)

    def _load_general_sets(self, mmap: xlrd.book.Book, set_name: str):
        """ Load sets from Map to the model according to the given set name. Usable for FUEL, EMISSION and TECHNOLOGY
        sets.

        Args:
            mmap (xlrd.book.Book): Model map file streamer.
            set_name (str): Set name. Maybe Fuel, Emission or Technology.

        Raises:
            MapProcessingException: Invalid set name

        """
        if set_name.upper() not in ["FUEL", "EMISSION", "TECHNOLOGY"]:
            raise MapProcessingException("set %s cannot be read using this method! Only Fuels, Emissions and "
                                         "Technologies are valid" % set_name)

        ws = mmap.sheet_by_name(set_name)
        hh = self._get_headers(ws, header_index=1)
        set_col = hh.index(set_name)

        # read and remove white spaces and repetitions
        elements = ws.col_values(set_col)[2:]
        elements = [x for x in list(set(elements)) if x != '']

        # Attribute to set
        self.sets[set_name.upper()].append(elements)

        return

    def _load_scenario_sets(self, mmap: xlrd.book.Book):
        """Reads REGION, YEAR, MODE_OF_OPERATION sets from the map and load to the model.

        Args:
            mmap (xlrd.book.Book): Model map file streamer.

        Raises:
            MapProcessingException: Scenario or Time setting not defined in the map.

        """
        ws = mmap.sheet_by_name("Scenario Settings")
        hh = self._get_headers(ws, header_index=1)

        # read Scenario
        scenario_col = hh.index("Scenario")
        try:
            scenario_row = list(ws.col_values(scenario_col)[:]).index(self.scenario)
        except ValueError:
            raise MapProcessingException("Model Scenario %s not defined in the Model Map" % self.scenario)

        # Read Time Setting
        self.time_setting = ws.cell_value(scenario_row, hh.index("Time Setting"))
        if self.time_setting == "":
            raise MapProcessingException("Time setting not defined for Scenario %s" % self.scenario)

        # Read sets
        for set_name in ["Region", "Year", "Mode_of_Operation"]:
            set_col = hh.index(set_name)
            # read cell value
            data = ws.cell_value(scenario_row, set_col)
            # Split multiple entries and remove spaces, (must be ; separated)
            if isinstance(data, str):
                data = data.replace(" ", "")
                data = data.split(";")
            self.sets[set_name.upper()].append(data)

    def _get_params_by_attr(self, attr_name, attr) -> Dict[str, OsemosysParam]:
        """
        Return a dict with all parameters that share the same attribute.

        Example:
            _get_params_by_attr("group", "emission") -> Return all parameters that are from the group emission.
        Args:
            attr_name (str): Attribute name
            attr: Attribute value

        Returns:
            Dict[str, OsemosysParam]: Dictionary with all parameters that share the same attribute.
                Keys are the parameter names.

        """
        params = dict()
        for p in self.param_names:
            param = self.params[p]
            if getattr(param, attr_name) == attr:
                params[p] = param
        if len(params) == 0:
            raise MapProcessingException("No param was found in the model with attribute %s equal to %s" %(attr_name, attr))
        return params

    def _load_params_in_sheet(self, mmap: xlrd.book.Book, sheet_name: str):
        """
        read all parameters in the sheet and load it to the Model.

        Args:
            mmap (xlrd.book.Book): Model map file streamer.
            sheet_name: Map Sheet from where the parameters will be read.

        Raises:
            MapProcessingException: Could not read the parameter. Missing index information for the parameter

        """
        ws = mmap.sheet_by_name(sheet_name)
        hh = self._get_headers(ws, header_index=1)

        # Set columns
        cc = self._get_dict_with_col_indexes_for_sets(hh)
        in_sheet_params = [x for x in hh if x in self.param_names]

        # for each param
        for p in in_sheet_params:
            sets = self.params[p].set_names
            # Special parameters that relate with Fuel input or output
            if self.params[p].type in ["FUEL_IN", "FUEL_OUT"]:
                sets[sets.index("FUEL")] = self.params[p].type

            # verify sets -> must be defined
            if any(x not in cc.keys() for x in sets):
                raise MapProcessingException("Parameter is %s is defined in Sheet %s, but index information given not "
                                             "enough" % (p, sheet_name))

            # Iterate lines
            param_col = ws.col_values(hh.index(p))
            for r in range(2, ws.nrows):
                val = param_col[r]
                if val == "":  # Empty --> Do nothing
                    pass
                elif "YEAR" in sets:  # year dependent -> Split function
                    # Assumed only one value for all year
                    if isinstance(val, float):
                        for y in self.sets["YEAR"].elm:
                            keys = [ws.cell_value(r, cc[x]) if x != "YEAR" else y for x in sets]
                            self.params[p].add_value(keys, val)

                    else:
                        # Assumed format YEAR Val;YEAR Val; YEAR Val ..
                        int_fun = _get_interpolation_function(val)
                        for y in self.sets["YEAR"].elm:
                            keys = [ws.cell_value(r, cc[x]) if x != "YEAR" else y for x in sets]
                            self.params[p].add_value(keys, int_fun(float(y)))

                else:
                    keys = [ws.cell_value(r, cc[x]) for x in sets]
                    self.params[p].add_value(keys, val)

    @staticmethod
    def _get_headers(ws: xlrd.sheet.Sheet, direction="x", header_index=0) -> List:
        """
        Function to get the header of an excel worksheet.
        Args:
            ws: excel worksheet to be read
            direction (:obj: str, optional):  which headers will be read. If x column headers are returned
                if "y" line headers are returned. The default is "x".
            header_index(:obj: int, optional): Which row/column are the header located. Default 0.

        Returns:
            List: List with the headers.
        """
        headers = []
        if direction == "x":
            for i in range(ws.ncols):
                headers.append(ws.cell_value(header_index, i))
        elif direction == "y":
            for i in range(ws.nrows):
                headers.append(ws.cell_value(i, header_index))
        return headers

    @staticmethod
    def _get_col_index(headers: List[str], val: str):
        try:
            col = headers.index(val)
        except ValueError:
            col = None
        return col

    @staticmethod
    def _get_dict_with_col_indexes_for_sets(hh):
        # Set columns
        cc = {'EMISSION': OsemosysInputGenerator._get_col_index(hh, "Emission"),
              'REGION': OsemosysInputGenerator._get_col_index(hh, "Region"),
              'TECHNOLOGY': OsemosysInputGenerator._get_col_index(hh, "Technology"),
              'FUEL': OsemosysInputGenerator._get_col_index(hh, "Fuel"),
              'FUEL_IN': OsemosysInputGenerator._get_col_index(hh, "Input Fuel"),
              'FUEL_OUT': OsemosysInputGenerator._get_col_index(hh, "Output Fuel"),
              'MODE_OF_OPERATION': OsemosysInputGenerator._get_col_index(hh, "Mode of Operation"), "YEAR": "NaN"}
        # Remove Keys with None
        return {k: v for k, v in cc.items() if v is not None}

    def write_model(self, filename="input.txt"):
        """
        Write model to osemosys input file  at the simulation directory.

        Args:
            filename (str, optional): Filename. Default "input.txt"

        """

        self.input_file = os.sep.join([self.dat_folder, filename])
        file = open(self.input_file, 'w')

        file.write('#############SETS#######################\n\n')
        for s in self.sets.keys():
            self.sets[s].write(file)

        file.write('#############PARAMETERS''#################\n\n')
        for s in self.params.keys():
            self.params[s].write(file)

    @classmethod
    def build_load_map_and_write(cls, name: str, scenario: str, data_dir: str, sim_dir: str):
        """
        Construct Model, load map and write input file.

        Args:
            name (str): Model name
            scenario (str): Scenario name
            data_dir (str): Path to Map file
            sim_dir (str): Path to simulation dir

        Returns:
            OsemosysInputGenerator: Model with all parameters and set defined.
        """
        input_gen = cls(name=name, scenario=scenario, data_dir=data_dir, sim_dir=sim_dir)
        print("Reading Model Map from: %s.." % data_dir)
        input_gen.read_model_map()
        print("Writing input file to folder: %s " % sim_dir)
        input_gen.write_model()
        print("INPUT GENERATION COMPLETED!")

        return input_gen

    @classmethod
    def build_from_input_file(cls, sim_dir, filename="input.txt"):
        """
        Construct model from an input file. Useful for model post processing.

        Args:
            sim_dir: Path to simulation dir. Model name and scenario read from simulation dir name.
            filename (str, optional): Input filename. Default "input.txt"

        Returns:
            OsemosysInputGenerator: Model with all parameters and set defined.
        """
        # take last part of simulation dir path
        sim_name = sim_dir.split(os.sep)[-1]
        name, scenario = tuple(sim_name.split("_"))
        model = cls(scenario=scenario, name=name, data_dir=None, sim_dir=sim_dir)

        fpath = os.sep.join([sim_dir, filename])
        model._read_input_file(fpath)
        return model

    def _read_input_file(self, file):
        """Method for building the model from an previous input file.

        Args:
            file (:obj: str, optional): Input file name. Default  "input.txt"

        """
        with open(file) as f:
            all_text = f.read()
            # remove comments and empty lines
            all_lines = all_text.split("\n")
            all_lines = [x for x in all_lines if len(x) > 0]
            all_lines = [x for x in all_lines if x[0] != '#']

            entries = " ".join(all_lines)
            all_lines.clear()

            entries = entries.split(";")

            # start reading sets and parameters!
            while entries:
                item = entries.pop(0)
                if "set" in item.lower():
                    self._read_set_entry(item)

                elif "param" in item.lower():
                    self._read_param_entry(item)

                elif item != '':
                    print("%s Entry ignored" % item)

    def _read_set_entry(self, entry):
        """Method for reading a set entry from previous input file"""
        header, values = entry.split(":=")
        ii = header.split(" ")
        set_i = ii.index("set") + 1
        set_name = ii[set_i]

        dim = self.sets[set_name].dim

        vv = values.split(" ")
        vv = [x for x in vv if x != ""]
        self.sets[set_name].append(vv)
        return

    def _read_param_entry(self, entry):
        """Method for reading a parameter entry from previous input file"""
        header, values = entry.split(":=")
        ii = header.split(" ")
        param_i = ii.index("param") + 1
        param_name = ii[param_i]

        # read default value id defined
        if "default" in header:
            def_i = ii.index("default") + 1
            self.params[param_name].default = float(ii[def_i])

        dim = self.params[param_name].dim

        vv = values.split(" ")

        vv = [x for x in vv if x != ""]

        while vv:
            keys = [vv.pop(0) for x in range(dim)]
            val = float(vv.pop(0))
            self.params[param_name].add_value(keys, val)

    def _calculate_year_multipliers(self):
        years = self.sets["YEAR"].elm
        for y in years:
            ii = years.index(y)
            if ii == 0:
                val = 1
            else:
                val = years[ii] - years[ii - 1]
            self.params["YearMultiplier"].add_value([y], val)
        pass


if __name__ == '__main__':
    data_dir = r"../data"
    a = OsemosysInputGenerator("DE-Model", scenario="test", data_dir=data_dir, sim_dir=data_dir)
    a.read_model_map()
    a.write_model()
    # a = OsemosysInputGenerator.build_from_input_file(sim_dir=r"..\runs\DE-Model_test")
    print("end")
