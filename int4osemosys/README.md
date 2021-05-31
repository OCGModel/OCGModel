# OCGModel Osemosys implementation

Thank you for your interest on the OCGModel.

This folder contains the comple source-code of the developed interface for OSeMOSYS and the model implementation using this interface.

This is organized as follows:
  - data: Folder with the models Technolgy maps (excel) and timeseries data
  - core: Where the interface source-code is located
  - scripts: Scripts for pre-processing data
  - OSeMOSYS: Folder with the adapated version of the OSeMOSYS code. All changes to the original code, are marked on the file. 
  
Note: You can also download the lattest version of the GNU MathProg Osemosys code and add to this folder to run with different Osemosys code versions. 
  

## 1. Installation

1. Install [GNU Linear Programming Kit](https://www.gnu.org/software/glpk/).

	At Linux systems this can be done by: 
		
		$ sudo apt-get install GLPK

	At Windows:
		
		- First download GLPK lattest version at 
		- Unzip the downloded folder and copy the glpk-4.**XX** folder to a directory (recommended: "C:\" )
		- Tell windows where to find GLPK, by adding the path the system path variables. 
		Be aware to use the version corresponding to your system type, i.e. "C:\glpk-4.**XX**\w32" for 32-Bit system or "C:\glpk-4.**XX**\w64" for 64-Bit system. 

	Test your isntallation by running the following line at the terminal: 
	
	$ glpsol --version
	
2. Install Python
	
	- If you do not have Python installed in your machine, you can download it [here](https://www.python.org/downloads/)

## 2. Run the model

1. **Run the startOsemosysInt.py script in this folder.** This can be done by following command:

	python startOsemosysInt.py 

2. Select the option "r"
3. Write the model and scenario name:
	
	Model name: ***OCGModel***
	Model name: ***base***
	

	




	
	



