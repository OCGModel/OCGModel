"""
Module to run the simulation.

(1) Calls Input Processing to build the input file model from the map.
(2) Creates simulation folder in the runs directory
(3) Calls solvers according to configuration file
(4) Save output in csv format. If Solver is CPLEX then the CPLEXSolutionProcessor is called to save selected variables to .csv

Example:
    >>> from core.ModelRunner import ModelRunner
    >>> from core.utils import settings

    Build engine and run:

    >>> run_engine = ModelRunner(model="Germany",scenario="base", sett=settings)
    >>> run_engine.run()

    or

    >>> ModelRunner.build_and_run(model="Germany",scenario="base", sett=settings)
"""

import os
import subprocess  # Run GLPSOL
import shutil  # copy files

from core.Settings import cplex  # cplex API if loaded

from core.InputProcessing import OsemosysInputGenerator
from core.OutputProcessing import CPLEXSolutionProcessor
from core.Settings import Settings


class ModelRunnerException(Exception):
    """
    Class for throwing Exceptions during the simulation running
    """
    pass


class ModelRunner:
    """
    Model Runner

    Args:
        model(str): Model name
        scenario (str): Scenario name
        sett (Settings): Settings object with the simulation settings
    """

    def __init__(self, model, scenario, sett: Settings):

        sim_dir = os.sep.join([sett.runs_dir, model + "_" + scenario])
        self.sett = sett
        self._make_dirs(sim_dir)

        self.scenario = scenario
        self.sim_dir = sim_dir
        self.model = model

        # copy Settings File to sim_folder --> post processing
        shutil.copyfile(sett.settings_file, os.sep.join([sim_dir, sett.filename]))

    def run(self):
        """
        Runs the model.

        Performs all steps necessary to run the model.
        (1) Calls Input Processing to build the input file model from the map.
        (2) Creates simulation folder in the runs directory
        (3) Calls solvers according to configuration file
        (4) Save output in csv format. If Solver is CPLEX then the CPLEXSolutionProcessor is called to save
        selected variables to .csv

        Raises:
            ModelRunnerException: Invalid input file format or generation of CPLEX input file failed.

        """
        model = self.model
        scenario = self.scenario
        sim_dir = self.sim_dir

        self._run_model(model, scenario, sim_dir)

    def _make_dirs(self, sim_dir: str):
        """
        Set up simulation directories depending on the selected solver
        Args:
            sim_dir(str): simulation directory
        """
        # make dir in runs folder
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)

        if self.sett.solver == "CPLEX":
            res_dir = os.sep.join([sim_dir, self.sett.cplex_solutions_dir])
        elif self.sett.solver == "GLPK":
            res_dir = os.sep.join([sim_dir, self.sett.glpk_solutions_dir])
        else:
            raise ModelRunnerException("Invalid settings for solver! Must be GLPK or CPLEX")

        # If exists -> must be emptied
        if os.path.exists(res_dir):
            for f in os.listdir(res_dir):
                os.remove(os.sep.join([res_dir,f]))
        else:
            os.makedirs(res_dir)

    def _run_model(self, model, scenario, sim_dir: str):

        sett = self.sett
        # Call Input Processing
        OsemosysInputGenerator.build_load_map_and_write(name=model, scenario=scenario, data_dir=sett.data_dir,
                                               sim_dir=sim_dir)

        # Copy osemosys code to sim dir
        source = os.sep.join([sett.osemosys_dir, sett.osemosys_code_version])
        dest = os.sep.join([sim_dir, sett.osemosys_code_version])
        shutil.copyfile(source, dest)

        # Run model
        if sett.solver == "GLPK":
            subprocess.call(["glpsol", "-m", sett.osemosys_code_version, "-d", "input.txt", "-o", "res.csv"],
                            cwd=sim_dir)
        if sett.solver == "CPLEX":
            if cplex == False:
                raise ModelRunnerException("Solver set to CPLEX, however API module not found!")
            # Generate input file
            f_type = sett.cplex_input_filename.split(".")[-1]
            if f_type not in ["lp"]:
                raise ModelRunnerException("%s Invalid CPLEX input file type! Format must be .lp"
                                           % sett.cplex_input_filename)

            rs = subprocess.call(["glpsol", "-m", sett.osemosys_code_version, "-d", "input.txt", "--w%s" % f_type,
                                  sett.cplex_input_filename, "--check"], cwd=sim_dir)
            if rs:
                raise ModelRunnerException("Generation of CPLEX input file failed!")

            # Run CPLEX
            # Initialize API
            cpx = cplex.Cplex()
            cpx.read(os.sep.join([sim_dir, sett.cplex_input_filename]))
            cpx.parameters.barrier.convergetol.set(float(sett.barepcomp))
            cpx.solve()

            # Write .sol file
            if sett.cplex_output_filename is not None:
                file = os.sep.join([sim_dir, sett.cplex_output_filename])
                # check and delete previous solution file in folder
                if sett.cplex_output_filename in os.listdir(sim_dir):
                    os.remove(file)
                # write
                cpx.solution.write(file)

            # save csv files
            cplex_post_processing = CPLEXSolutionProcessor(cpx)
            cplex_post_processing.save_main_vars_to_csv(os.sep.join([sim_dir, sett.cplex_solutions_dir]))

    @classmethod
    def build_and_run(cls,model, scenario, sett: Settings):
        """
        Initialize the runner engine and call the run method.

        Args:
            model(str): Model name
            scenario (str): Scenario name
            sett (Settings): Settings object with the simulation settings.

        Returns:
            ModelRunner: Model runner engine

        """
        model_runner = ModelRunner(model, scenario, sett)
        model_runner.run()

        return model_runner

