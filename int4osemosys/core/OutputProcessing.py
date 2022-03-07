"""
Module for post processing
"""

from core.InputProcessing import OsemosysInputGenerator
from core.Settings import Y, T, L, F, E, M, R, L, LS, LD, LH, S
from core.Settings import cplex

import os
import warnings
import pandas as pd
import numpy as np
import itertools


class OutputProcessingException(Exception):
    """
    Class for throwing Exceptions during post processing
    """
    pass

class CPLEXSolutionProcessingException(OutputProcessingException):
    """
    Class for throwing Exceptions during post processing with CPLEX python API
    """
    pass


class SolutionInterface:
    """ Class for post processing of the model solution

    Args:
        sol_dir(str): Directory where the solution is found.

    Raises:
        OutputProcessingException: Error during results post processing

    """
    def __init__(self, sol_dir):
        self.sol_dir = sol_dir

    def compute_all_variables(self, input_generator, vars_to_compute= None):
        """Compute intermediate variables from main variables.

        Compute intermediate variables from main by using the model parameters set in the the input_generator object.
        As not all Osemosys code version contains the intermediate variables, this methods computes them and also add
        the respective csv file to the solution directory. The computed variables are:

        >>> vars_to_compute = ["AnnualEmissions", "TotalTechnologyAnnualActivity", "UseByTechnology",
        >>>                    "UseByTechnologyAnnual", "ProductionByTechnologyAnnual", "ProductionByTechnology",
        >>>                   "Demand",  "RateOfUseByTechnology", "RateOfProductionByTechnology", "RateOfDemand",
        >>>                   "TotalCapacityAnnual" ]

        Args:
            input_generator(OsemosysInputGenerator): OSEMOSYS Model with all parameters values

        """
        if vars_to_compute is None:
            vars_to_compute = ["AnnualEmissions",
                           "TotalTechnologyAnnualActivity",
                           "UseByTechnology", "UseByTechnologyAnnual",
                           "ProductionByTechnologyAnnual", "ProductionByTechnology",
                           "Demand",
                           "RateOfUseByTechnology", "RateOfProductionByTechnology", "RateOfDemand",
                           "TotalCapacityAnnual"]


        for var in vars_to_compute:
            var_filename = os.sep.join([self.sol_dir, var + ".csv"])
            if not os.path.isfile(var_filename):
                self.compute_variable(var, input_generator)

        print( "All variables computed")


    def get_var_values(self, var, select=None, return_zeros=False, compute_missing_vars = False, input_generator=None) -> pd.DataFrame:

        """Get the value of a variable.

        Args:
            var(str): Variable name
            select (dict[str,str], optional): Select index of variables to be returned. Dictionary where the keys are
                                              the set names and the entries the index to be selected.  Defaults to None.

            return_zeros (bool, optional): Indicator weather variables with zero as values should be selected. Defaults to False.

            compute_missing_vars(bool, optional): If true and the variable ".csv" is not found the method
                                                  compute_variable is called.
                                                  Requires input_generator to be defined. Defaults to False

            input_generator (OsemosysInputGenerator, optional): OSEMOSYS Model with all parameters values. Defaults to
                                                                None. If compute_missing_vars is True, this must be
                                                                given otherwise an error is raised.

        Returns:
            pandas.DataFrame: Dataframe with the indexes and variable values.
        """

        var_file = os.sep.join([self.sol_dir, var + ".csv"])
        try:
            df = pd.read_csv(var_file)
        except FileNotFoundError:
            if not compute_missing_vars:
                raise OutputProcessingException("Variable %s csv file not found." % var)
            elif input_generator is not None:
                try:
                    print("Variable %s not found. Trying to compute from main variables..." % var)
                    self.compute_variable(var, input_generator=input_generator)
                    df = pd.read_csv(var_file)
                except OutputProcessingException as e:
                    raise e
            else:
                raise OutputProcessingException("Input Generator not Given! Could not process variable")

        if select is not None:
            for k in select.keys():
                if k in df.columns:
                    # Select
                    df = df[df[k] == select[k]]
                else:
                    warnings.warn("%s is not a variable index for var %s. Selection ignored!" % (k, var))

        if not return_zeros:
            df = df[df["VALUE"] != 0]

        return df

    def compute_variable(self, varname, input_generator: OsemosysInputGenerator):
        """
        Compute intermediate variable from main variables

        As not all Osemosys code version contains the intermediate variables, this methods computes the varibale with
        name "varname" and also add the respective csv file to the solution directory.

        Args:
            varname(str): Variable name to be computed
            input_generator(OsemosysInputGenerator): OSEMOSYS Model with all parameters values

        """


        base_var = "RateOfActivity"
        params_divide = []

        if varname == "AnnualEmissions":
            param_names = ["YearSplit", "EmissionActivityRatio"]
            sets = [R, E, Y]

        elif varname == "TotalTechnologyAnnualActivity":
            param_names = ["YearSplit"]
            sets = [R, T, Y]

        elif varname == "UseByTechnology":
            param_names = ["InputActivityRatio", "YearSplit"]
            sets = [R, L, T, F, Y]

        elif varname == "UseByTechnologyAnnual":
            param_names = ["InputActivityRatio", "YearSplit"]
            sets = [R, T, F, Y]

        elif varname == "ProductionByTechnology":
            param_names = ["OutputActivityRatio", "YearSplit"]
            sets = [R, L, T, F, Y]

        elif varname == "ProductionByTechnologyAnnual":
            param_names = ["OutputActivityRatio", "YearSplit"]
            sets = [R, T, F, Y]

        elif varname == "RateOfProductionByTechnology":
            param_names = ["OutputActivityRatio"]
            sets = [R, L, T, F, Y]

        elif varname == "RateOfUseByTechnology":
            param_names = ["InputActivityRatio"]
            sets = [R, L, T, F, Y]

        elif varname == "RateOfDemand":
            param_names = ["SpecifiedAnnualDemand", "SpecifiedDemandProfile"]
            params_divide = ["YearSplit"]
            base_var = None
            sets = [R, L, F, Y]

        elif varname == "Demand":
            param_names = ["SpecifiedAnnualDemand", "SpecifiedDemandProfile"]
            base_var = None
            sets = [R, L, F, Y]

        elif varname == "TotalCapacityAnnual":
            # special case!
            self._compute_annual_active_capacity(input_generator=input_generator)
            return
        else:
            raise OutputProcessingException("Invalid Variable name: %s" % varname)

        # Load base Var
        if base_var is not None:
            base_var_file = os.sep.join([self.sol_dir, base_var + ".csv"])
            base_var = pd.read_csv(base_var_file)

        # Calculate and save new CSV
        df = self._multiply(base_var, input_generator, param_names, params_divide)
        df = df.groupby(sets).sum()["VALUE"]
        df.to_csv(os.sep.join([self.sol_dir, varname + ".csv"]))

        print("%s Exported to CSV!" % varname)

    def _compute_annual_active_capacity(self, input_generator: OsemosysInputGenerator):
        ret_dict = dict()
        varname = "TotalCapacityAnnual"
        new_cap_file = os.sep.join([self.sol_dir, "NewCapacity" + ".csv"])
        new_cap = pd.read_csv(new_cap_file)
        new_cap = new_cap.groupby([R, T, Y]).sum()["VALUE"]

        sets = [R, T, Y]
        set_to_iterate = [input_generator.sets[x].elm for x in sets]

        i = 0
        for (r, t, y) in itertools.product(*set_to_iterate):
            # initiate dict
            ret_dict[i] = {R: r, T: t, Y: y}

            op_life = input_generator.params["OperationalLife"][r, t]
            cap = input_generator.params["ResidualCapacity"][r, t, y]
            # Calculate active cap
            for yy in input_generator.sets[Y].elm:
                if 0 <= y - yy < op_life:
                    try:
                        cap = cap + new_cap[r, t, yy]
                    except KeyError:
                        pass
            ret_dict[i]["VALUE"] = cap
            i = i + 1

        df = pd.DataFrame.from_dict(ret_dict, orient="index")
        df.to_csv(os.sep.join([self.sol_dir, varname + ".csv"]))
        print("%s Exported to CSV!" % varname)

    @staticmethod
    def _multiply(base_var, input_generator, param_names, params_divide=None):

        if params_divide is None:
            params_divide = []

        # take all dim
        dims = []
        for pname in param_names + params_divide:
            p = input_generator.params[pname]
            dims = dims + [x for x in p.set_names if x not in dims]

        ret_dict = dict()
        if base_var is not None:
            base_var_dict = base_var.to_dict("index")
            sets_to_iterate = [x for x in dims if x not in base_var.columns]
        else:
            base_var_dict = {0: {"VALUE": 1}}
            sets_to_iterate = dims

        set_elm_to_iterate = [input_generator.sets[x].elm for x in sets_to_iterate]

        i = 0
        for ii in itertools.product(*set_elm_to_iterate):
            # For each line
            for k in base_var_dict.keys():
                ret_dict[i] = base_var_dict[k].copy()

                # Add to return dict
                for s in sets_to_iterate:
                    ret_dict[i][s] = ii[sets_to_iterate.index(s)]

                for pname in param_names + params_divide:
                    p = input_generator.params[pname]
                    keys = [ret_dict[i][x] for x in p.set_names]
                    if ret_dict[i]["VALUE"] == 0:
                        break
                    elif pname in param_names:
                        ret_dict[i]["VALUE"] = ret_dict[i]["VALUE"] * p[keys]
                    elif pname in params_divide:
                        ret_dict[i]["VALUE"] = ret_dict[i]["VALUE"] / p[keys]
                i = i + 1

        df = pd.DataFrame.from_dict(ret_dict, orient="index")
        df = df[df["VALUE"] != 0]

        return df


class CPLEXSolutionProcessor:
    """
    Post Processing Engine for models solved with CPLEX.

    Args:
        c(cplex.Cplex): Cplex object with the solved model

    Raises:
        CPLEXSolutionProcessingException: Error while post processing with CPLEX API

    Attributes:

        main_vars (dict[str,list[str]]): Dictionary with variable name as key and respective indexing sets as
                                        list, e.g. ["FUEL", "YEAR"].
                                        "Main" Variables are the ones present at all
                                        OSEMOSYS code versions and that cannot be computed from other variables.

    """
    def __init__(self, c: cplex, model_generator=None):

        if cplex == False:
            raise CPLEXSolutionProcessingException("Trying to use CPLEX solver, but the same could not be found!")

        self.c = c

        self.main_vars = dict(AnnualEmissions=[R, E, Y],
                              Demand=[R, L, F, Y],
                              TotalCapacityAnnual=[R, T, Y],
                              TotalTechnologyAnnualActivity=[R, T, Y],
                              UseByTechnology=[R, L, T, F, Y],
                              ProductionByTechnology=[R, L, T, F, Y],
                              RateOfActivity=[R, L, T, M, Y],
                              ProductionByTechnologyAnnual=[R, T, F, Y],
                              UseByTechnologyAnnual=[R, T, F, Y],
                              RateOfProductionByTechnology=[R, L, T, F, Y],
                              RateOfUseByTechnology=[R, L, T, F, Y],
                              RateOfDemand=[R, L, F, Y],
                              NewCapacity=[R, T, Y],
                              DiscountedSalvageValue=[R, T, Y])

        pass

    def save_main_vars_to_csv(self, save_folder):
        """Save variables to csv.

        Each variable is saved to a different csv file named after the variable name.

        Args:
            save_folder (str): Folder where the csv files will be saved

        Returns:

        """
        # verify if problem solved!
        if not self.c.solution.get_status():
            raise CPLEXSolutionProcessingException("CPLEX Problem not solved!")

        var_names = np.array(self.c.variables.get_names())
        all_values = np.array(self.c.solution.get_values())
        for v in self.main_vars.keys():
            print("Exporting %s to csv..." % v)
            # find variable name
            ii = [v == x.split("(")[0] for x in var_names]
            if any(ii):
                df = pd.DataFrame(dict(keys=var_names[ii], VALUE=all_values[ii]))

                # Correct indexing
                df = self._correct_indexing_df(df, column_names=self.main_vars[v])
                df.to_csv(os.sep.join(["%s" % save_folder, "%s.csv" % v]))

            else:
                warnings.warn("Variable %s was not computed in the model."
                              "Use the solutionInterface to compute intermediate variables" % v)

        print("Main variables exported to csv")

    @classmethod
    def _correct_indexing_df(cls, df, dim=None, column_names=None, key_column="keys") -> pd.DataFrame:
        if dim is None and column_names is None:
            raise OutputProcessingException("Key dimension or column names must be given")
        elif dim is None and column_names is not None:
            dim = len(column_names)
        else:
            column_names = []
            for i in range(dim):
                column_names.append("i%d" % i)

        # Apply correction
        aux = df[key_column].apply(lambda x: cls._list_key_from_str(x.split("(")[1]))
        for i in range(dim):
            col = column_names[i]
            df.loc[:, col] = aux.apply(lambda x: x[i])

        return df

    @staticmethod
    def _list_key_from_str(key: str) -> list:
        # key format (XX, XX, XX; XX)
        # remove parenthesis
        key = key.replace("(", "").replace(")", "")
        list_key = key.split(",")
        return list(list_key)



