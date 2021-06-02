.. _osemosysimp:

Osemosys Implementation
=======================

The OCG Model is implemented using the Open Source Energy Modelling System (OSeMOSYS) through an.
See :ref:`Osemosys Interface Documentation<intdoc>`.

.. note::

   Learn more about OSeMOSYS at `http://www.osemosys.org <http://www.osemosys.org>`_

1. Install
------
**Download the OCG Model Repository**
The OCG Model can be downloaded from our `GitHub repository <https://github.com/OCGModel/OCGModel.git>`_

**Install GLPK** The GNU Linear Programming Kit (GLPK) is a open

   The following call should install GLPK at **Linux** systems:

   .. code-block::

      $ sudo apt-get install glpk

   For **Windows** users:

      * Download the latest version of GLPK at `https://sourceforge.net/projects/winglpk/ <https://sourceforge.net/projects/winglpk/>`_
      * Unzip the download file *winglpk-4.65* and locate the folder *glpk-4.65* inside.
      * Copy the whole folder *glpk-4.65*  to your C:\ directory.

      With these steps GLPK is installed, now you have to let windows know where to find GLPK and add the folder to the paths variables.
      For that:

      * Navigate to *Control Panel > System > About* and check the system type (x32 or x64).
      * Open your *environment variables*.
         * Type Win+r, type *systempropertiesadvanced* and press ok.

         .. figure:: /images/winexecute.jpg
            :align: center
            :scale: 100%

         * At the new window select "Environmental variables"

      


**Install CPLEX API** optional. The implemented interface can also work with the the CPLEX solver.

Edit the model or create a new one.
------------------------------------
Techmap

Model: Germany
Scenario: 80ClimateGoal

Run the model
---------------
After the model data is given/ edited in the respective map file start the interface.
For that run the startOsemosysInt.py script in your API or through the following command:

.. code-block:: bash

   $ python startOsemosysInt.py

The Following menu will appear:

.. code-block:: shell
   :emphasize-lines: 4

   ...reading Settings from settings.ini
   ####Interface for Osemosys####
   Select option:
   r - Run model
   p - Plot results
   q - Quit

Type **"r"** to run the model.
You will then be asked for the model name as scenario.

.. code-block:: bash

   Model name: $ Germany
   Scenario: $ 80ClimateGoal

If the model and scenario name were given correctly, the mo will and the solver log will be displayed.



Run.

You can change settings (Osemosys code version, solver, etc...) in the ..
In the under the runs folder in the directory a folder will the model name and scenario name.
The results files can be found in csv form in the resut under folder.


Plot results
------------

For the analyse of the results, the interface also offers  an plotting engine.
To plot the results select **'p'** in the main menu.

.. code-block:: shell
   :emphasize-lines: 4

   ####Interface for Osemosys####
   Select option:
   r - Run model
   p - Plot results
   q - Quit

A list with all simulations  in the runs folder will be displayed and can be selected.
Once selected the respective simulation, a list of all plots will be displayed.

As example
Iterative ploting code.

.. note::
   See the  :ref:`plotting engine documentation<plotting>` for a short description of each plot.
