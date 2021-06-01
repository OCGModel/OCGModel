#!/usr/bin/env python
#import sys

from core.InputProcessing import MapProcessingException
from core.ModelRunner import ModelRunner
from core.ResPlotting import OsemosysPlottingEngine, ResPlottingException
from core.Settings import Settings, SETTINGS_FILENAME

from inspect import signature
import os


print("...reading Settings from settings.ini \n")
print("####Interface for Osemosys####")
sett = Settings(SETTINGS_FILENAME)
opt = ""
while opt != "q":
    opt = input("\nSelect option:\n\n "
                "r - Run model \n "
                "p - Plot results\n "
                "s - Settings\n "
                "q - Quit\n"
                ">> ").lower()
    if opt == "r":
        while True:
            model = input("Model name: ")
            scenario = input("Scenario: ")
            if model == "q" or scenario == "q":
                print("Quiting..")
                break
            try:
                mr = ModelRunner.build_and_run(model=model, scenario=scenario, sett=sett)
                break
            except MapProcessingException as e:
                print(e)
    elif opt == "p":
        print("\nSelect Simulation to plot\n")
        sim = ""
        simulations = os.listdir(sett.runs_dir)
        print("Simulations: \n")
        for i in range(len(simulations)):
            print("% i - %s" % (i, simulations[i]))

        # Select simulation to be computed
        valid_option = False
        while not valid_option:
            try:
                s = int(input(">> "))
                sim_dir = os.sep.join([sett.runs_dir, simulations[s]])
                valid_option = True
            except ValueError:
                print("Invalid option!")

        print("Starting plotting engine for %s " % simulations[s])
        if not sett.compute_missing_variables:
            pe = OsemosysPlottingEngine(sim_dir)
        else:
            pe = OsemosysPlottingEngine(sim_dir, compute_missing_variables=True)

        pe_opt = ""
        while pe_opt != "q":
            plots = [x for x in dir(pe) if x[0] != "_" and "plot" in x]
            print("\nSelect Plot: \n ")
            for j in range(len(plots)):
                print("%i - %s" % (j, plots[j]))
            pe_opt = input(">> ")
            try:
                int_opt = int(pe_opt)
                plot_fun = getattr(pe, plots[int_opt])
                print("Please give the inputs for %s. [help: %s]" % (plots[int_opt], str(signature(plot_fun))))
                params = []
                sig = str(signature(plot_fun)).replace("(", "")
                sig = sig.replace(")", "")
                sig = sig.replace(" ", "")
                sig = sig.split(",")
                if sig != ['']:
                    for el in sig:
                        params.append(input("%s: " % el))
                plot_fun(*params)
                input("Results plotted. Press any key to continue")
            except (ValueError, ResPlottingException, IndexError) as e:
                if isinstance(e, ResPlottingException):
                    print(e)
                    input("Press any key to continue")
                elif isinstance(e, IndexError):
                    print("Invalid Option!")
                elif pe_opt == "q":
                    print("..returning to main menu.")
                else:
                    raise e
    elif opt == "s":
        c_options = ["osemosys_code_version", "solver", "compute_missing_variables", "sankey_opacity"]
        print("Which configuration do you want to edit?: ")
        for opt in c_options:
            print("%d - %s" %(c_options.index(opt), opt))
        c_opt = int(input())
        opt = c_options[c_opt]

        print("Current value of %s is %s" %(opt, getattr(sett, opt)))
        if opt in sett.comments.keys():
            print("%s" %sett.comments[opt])

        new_val = input("New value [enter to skip]: ")
        if new_val != "":
            setattr(sett, opt, new_val)
            sett.update()
            input("\nSettings file updated! This will affect only new model runs! \nPress any key to continue")
    elif opt == "q":
        print("..quitting Interface for Osemosys.")
        pass
    else:
        print("Invalid Option!! \n\n")
