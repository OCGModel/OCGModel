# OCGModel Osemosys implementation

Thank you for your interest on the OCGModel!

This folder contains the complete source-code of the developed interface for [OSeMOSYS](http://www.osemosys.org) and the model implementation using this interface.

It is organized as follows:
  - data: models Technolgy map (excel) and timeseries data
  - core: interface source-code is located
  - scripts: scripts for pre-processing data
  - OSeMOSYS: Folder with the adapated version of the OSeMOSYS code. All changes to the original code, are marked on the file. 
  
Note: You can also download the [lattest version of the GNU MathProg Osemosys code](http://www.osemosys.org/get-started.html) and add to the OSeMOSYS folder to run with the different code versions. 
  

## 1. Installation

1. Install [GNU Linear Programming Kit](https://www.gnu.org/software/glpk/).

	At Windows:
		
	- First download GLPK lattest version at [https://sourceforge.net/projects/winglpk](https://sourceforge.net/projects/winglpk/)
	- Unzip the downloded folder and copy the glpk-4.**XX** folder to a directory (recommended: "C:\" )
	- Tell windows where to find GLPK, by adding the path the system path variables. 
	Be aware to use the version corresponding to your system type, i.e. "C:\glpk-4.**XX**\w32" for 32-Bit system or "C:\glpk-4.**XX**\w64" for 64-Bit system. 

	Test your installation by running the following line at the terminal: 
	
		$ glpsol --version
		
	At Linux: 
		
		$ sudo apt-get install GLPK
		
2. Install Python (with pip)
	
	- If you do not have Python installed in your computer, you can download it [here](https://www.python.org/downloads/).
        Make sure to check **"Add Python to PATH"** at the installation assistant. If you opt for the customizable installation make sure to include pip.

3. Install required python packages

	At Windows:
	
	- Run the setup.bat file in this directory
	
	At Linux:
	
	Be sure open the terminal in this folder and the following call should install the packages: 
	
		$ pip install -r core/requirements.txt

## 2. Run the model

1. **Start Interface** 
 	- On Windows, you can simply run the ***start_int4osemosys.bat*** file.
 	- Or ***Run the startOsemosysInt.py script in this folder.*** This can be done using the prefered IDE or by the following command:

			$ python startOsemosysInt.py 

2. Select the option "r" -Run
3. Write the model and scenario name and press enter
	
		Model name: OCGModel
		Scenario: base

## 3. Plot Results

You can use the same interface to plot the model resuls. For that:

1. Select the option "p" -  Plot results at the main menu. 
2. Select the desired plot from the list
3. Give the additional information for the type of plot selected. e.g. Fuel, year, ..etc. **The paramerets must be given exactly how declared in the Techmap file!! CASE SENSITIVE!**
	Example:
		
		Select Plot:
		0 - plot_annual_active_capacities
		1 - plot_annual_emissions
		2 - plot_annual_supply
		3 - plot_annual_use
		4 - plot_sankey
		5 - plot_supply_timeseries
		6 - plot_use_timeseries
		
		>> 2
		
		Please give the inputs for plot_annual_supply. [help: (fuel)]
		fuel: Electricity
				
5. The plot will be displayed at your internet browser. For the example above, the following image is expected:

	<p align="center">
 	<img src = "..\docs\source\images\example_plot.png" width = 500 >
	</p>

	Note: You can also find the CSV files with the simulation results at ***.\run\OCGModel_base"***

## 4. Edit or create new model instance

All model instance parameters are located at the Technology Map at the folder ***data***. To edit the model, simply open the file with the model name and edit it there.
You can change existing parameters, create new scenarios, add fuels and technologies to the model. After editing simply save and close the file. 

To create a new model, you can duplicate the OCGModel technoly map file. Then, rename the file with the desired model name and edit it.
To run the new model, use the same name given to the technology map. 

## Need help? 

### Original Paper
You can find a complete description of the model structure,  assumptions and validation results at the [Original Paper](https://www.mdpi.com/1996-1073/14/23/8084)
	
### OCGModel and int4osemosys documentation
You can read our complete documentation [here](../docs/OCGModel_docs.pdf)
	
### OSeMOSYS documentation
You can read the OSeMOSYS original documentation [here](https://osemosys.readthedocs.io/en/latest/?badge=latest)

## Citing OCGModel

If you use the OCGModel in your research, please cite the original paper:

	Barbosa, J.; Ripp, C.; Steinke, F. Accessible Modeling of the German Energy Transition: An Open, Compact, and Validated Model. Energies 2021, 14, 8084. https://doi.org/10.3390/en14238084
For Bibtex you can use the following:
	
	@Article{en14238084,
	AUTHOR = {Barbosa, Julia and Ripp, Christopher and Steinke, Florian},
	TITLE = {Accessible Modeling of the German Energy Transition: An Open, Compact, and Validated Model},
	JOURNAL = {Energies},
	VOLUME = {14},
	YEAR = {2021},
	NUMBER = {23},
	ARTICLE-NUMBER = {8084},
	URL = {https://www.mdpi.com/1996-1073/14/23/8084},
	ISSN = {1996-1073},
	ABSTRACT = {We present an easily accessible model for dispatch and expansion planning of the German multi-modal energy system from today until 2050. The model can be used with low efforts while comparing favorably with historic data and other studies of future developments. More specifically, the model is based on a linear programming partial equilibrium framework and uses a compact set of technologies to ease the comprehension for new modelers. It contains all equations and parameters needed, with the data sources and model assumptions documented in detail. All code and data are openly accessible and usable. The model can reproduce today&rsquo;s energy mix and its CO2 emissions with deviations below 10%. The generated energy transition path, for an 80% CO2 reduction scenario until 2050, is consistent with leading studies on this topic. Our work thus summarizes the key insights of previous works and can serve as a validated and ready-to-use platform for other modelers to examine additional hypotheses.},
	DOI = {10.3390/en14238084}
	}






	
	



