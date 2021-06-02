.. _osemosysimp:

Osemosys Implementation
=======================

The OCG Model is implemented using the Open Source Energy Modelling System (OSeMOSYS) through the int4osemosys interface.
See :ref:`Osemosys Interface Documentation<intdoc>`.

.. note::

   Learn more about OSeMOSYS at `http://www.osemosys.org <http://www.osemosys.org>`_

   
Installation
------------

1. **Download the OCG Model Repository**
   
   The OCG Model can be downloaded from our `GitHub repository <https://github.com/OCGModel/OCGModel.git>`_

2. **Install** `GNU Linear Programming Kit <https://www.gnu.org/software/glpk/>`_.

   At Windows:
      
   - First download GLPK lattest version at `https://sourceforge.net/projects/winglpk/ <https://sourceforge.net/projects/winglpk/>`_
   - Unzip the download file *winglpk-4.XX* and locate the folder *glpk-4.XX* inside.
   - Copy the whole folder *glpk-4.XX*  to your "C:\" directory.
   - Navigate to *Control Panel > System > About* and check the system type (x32 or x64).
   - Type Win+r, type *systempropertiesadvanced* and press ok.
   - At the new window, select "Environmental variables" below right
   - Under the System variables, select "Path" and then click "Edit"
   - Select "New" and add the path to GLPK. Be aware to use the version corresponding to your system type, i.e. "C:\glpk-4.**XX**\w32" for 32-Bit system or "C:\glpk-4.**XX**\w64" for 64-Bit system. 

      Test your installation by running the following line at the terminal: 
      
      .. code-block:: bash
      
            $ glpsol --version
      
   At Linux the following call should install GLPK: 

   .. code-block:: bash      
      
      $ sudo apt-get install GLPK
      
3. **Install Python (with pip)**
   
   If you do not have Python installed in your computer, you can download it `here <https://www.python.org/downloads/>`_. 
   Make sure to check **"Add Python to PATH"** at the installation assistant. If you opt for the customizable installation make sure to include pip.

4. **Install required python packages**

   At Windows:
   
   Run as *administrator* the ***setup.bat*** file in the int4osemosys directory.
   
   At Linux:
   
   Be sure open the terminal in the int4osemosys directory and the following call should install the packages: 
   
      .. code-block:: bash
      
         $ pip install -r core/requirements.txt

5. (optional) Install CPLEX API

   If you have acess to CPLEX, the interface also offers support to the CPLEX solver.
   For that you need to  `install the CPLEX Python API <https://www.ibm.com/docs/en/icos/12.10.0?topic=cplex-setting-up-python-api>`_ and change the interface settings to CPLEX.
   See how to change the settings :ref:`here <intsett>`.

Run the model
-------------

To start the interface:

- On Windows, you can simply run the **start_int4osemosys.bat** file.
- Or **Run the startOsemosysInt.py** script in this folder. This can be done using the prefered IDE or by the following command:

      .. code-block:: shell
      
         $ python startOsemosysInt.py 


The Following menu will appear:

.. code-block:: shell
   :emphasize-lines: 8

   ...reading Settings from settings.ini
   ####Interface for Osemosys####
   Select option:
   r - Run model
   p - Plot results
   s - Settings
   q - Quit
   >> r


Type **"r"** to run the model.
You will then be asked for the model name as scenario.

.. code-block:: shell

   Model name: OCGModel
   Scenario: base


If the model and scenario name were given correctly, the solver log will be displayed.

Plot Results
------------

You can use the same interface to plot the model resuls. For that:
      
   1. Select the option "p" -  Plot results at the main menu. 
   2. A list with all simulations in the runs folder will be displayed and can be selected. 
   3. Give the additional information for the type of plot selected. e.g. Fuel, year,... **The paramerets must be given exactly how declared in the Techmap file!! CASE SENSITIVE!**
     
      Example:
         
      .. code-block:: shell
         :emphasize-lines: 9,11
      
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


      .. note::
         See the  :ref:`plotting engine documentation<plotting>` for a short description of each plot.

   4. The plot will be displayed at your internet browser. For the example above, the following image is expected:

      .. figure:: /images/example_plot.png
         :align: center
         :scale: 15%
      
               
   .. important:: You can also find the CSV files with the simulation results at **". \ run \ OCGModel_base"**


Edit Model
----------

The model instance parameters are located at the Technology Map **OCGModel.xlsx** at the folder **data**.
To edit the model, simply open the file with the model name and edit it there.
You can change existing parameters, create new scenarios, add fuels and technologies to the model. After editing save the file and run the model again.

To create a new model, you can duplicate the OCGModel technoly map file. Then, rename the file with the desired model name and edit it.
To run the new model, use the same name given to the technology map. 


.. important:: *time-slice dependent paramerets*: These parameters are given in the Technology map file by the name of the profile. Be sure to have at the **data\timeseries**
               folder, a .txt file with the profile name. The file should contain 8760 entries. See the examples provided in the folder. 


.. important:: *Time Setting*: The time setting will affect how the time profiles files are interpreted.
               You can use two different types of time settings: **mean** and **selection**. 
               
               For a **selection** time setting, a .csv file is required at the timeseries folder only with the selected timesteps, the profiles are obtained from the .txt files.
               
               For a **mean** time setting, a .csv file with the time setting name is also required. This file should contain not only the time-slices names, but also one column for each profile used in the model.
               
               By using the :ref:`script <tss>` made availble for selecting the timeslices, the output is alredy in the required format. 

.. _intsett:

Change Settings
---------------

You can change some settings by typing "s" on the main interface menu.
These are:

- *Osemosys code version*: 
   Version of the OSeMOSYS to be used to generate the model instance.

   Defaults to our adapted version (osemosys_short_OCG.txt) at the OSeMOSYS directory. All adpations to the original version are docummented in the file.  

   You can download the other versions of `OSeMOSYS <http://www.osemosys.org/get-started.html>`_, add to the the OSeMOSYS directory and change this setting to build the model instance with them. 

- *solver*:   
   Solver to be used. The interface offers support for the open-sorce GLPK and also CPLEX.

   If you have access to CPLEX and wish to use it, don't forget to install the cplex python API before changing this configuration.  

- *compute missing variables*: This parameter tell then if the not computed variables should be calcuated.

   The OSeMOSYS short code version does not compute intermediate variables, e.g ProductionByTechnology, UseBytechnology. 
      
   If True (1-default), the necessary variables will be computed on demand by the plotting engine. ( takes longer time to initilize the plotting engine).
   
   If False (0), trying to plot plots that required a non-computed variable will raise an error.
   

- *sankey opacity*:
   Link opacity on the Sankey plot. 


**(NOT RECOMMENDED)** You can also change the settings directlly on the **settings.ini** file located at the **core** directory. 
