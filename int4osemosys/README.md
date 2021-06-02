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

	Test your isntallation by running the following line at the terminal: 
	
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

1. **Strat Interface** 
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

## Need help? 

### Original Paper
You can find a complete description of the model structure,  assumptions and validation results at the [Original Paper](https://www.google.de)
	
### OCGModel and int4osemosys documentation
You can read our complete documentation [here](https://www.google.de)
	
### OSeMOSYS documentation
You can read the OSeMOSYS original documentation [here](https://osemosys.readthedocs.io/en/latest/?badge=latest)

## Citing OCGModel

If you use the OCGModel in your research, please cite the original paper:

	todo: addd paper name and citations 

For Bibtex you can use the following:
	
	@article{TODO: Add bibtex cite}






	
	



