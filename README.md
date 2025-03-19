# Code for A CHARACTERIZATION OF PATH-LENGTH MATRICES OF UNROOTED BINARY TREES

## General information
The code in this repository is written in Python 3. The library dependencies of the code are contained in the Conda environment file "solve310.yml".
You can manually install all the dependencies therein, or you can optionally import this file as a Conda environment with the command:
<br><br>
conda env create -f solve310.yml
<br><br>
from the Anaconda prompt. The optional Anaconda installation can be downloaded at https://www.anaconda.com/download.
<br>
The code uses DOCplex API for MILP. Academics and students can verify their access to a free version of CPLEX at https://www.ibm.com/academic/.

## Structure outline of the repository

|---> data <br>
&nbsp;&nbsp;&nbsp; |---> instances <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |---> original (problem instances used in Section 4 and Appendix A)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |---> buneman quadruplets (optimal values of y in F1-Buneman for each instance)<br>
|---> results (results presented in Section 4 and Appendix A)<br>
&nbsp;&nbsp;&nbsp; |---> BMEP_classic <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |---> F1-Manifold <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |---> F1-Buneman <br>
&nbsp;&nbsp;&nbsp; |---> BMEP_classic_contractions <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |---> Contractions <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |---> Contractions-Manifold <br>

## Support code for Proposition 12

- "combinations/generate_combinations.py" is the main script for generating the combinations of path-length sequences.<br>
- "config.py" is a configuration script containing values for the parameters "n" (i.e, the number of leaves) and "solver", which depends on the MILP solver installed on the runtime platform (e.g., "cplex" for IBM CPLEX and "xpress" for FICO Xpress, according to pyomo documentation: https://pyomo.readthedocs.io/en/stable/).

## Code used in Section 4, Appendix A, and Appendix C

- "main.py" is the main script for using F1-Manifold and F1-Buneman (both with and without disjunctive constraints).<br>
- "src/ini" is the folder containing the different configurations described in Section 4. The module "main.py" loads the "ini/main.ini" configuration. In order to load one or more different configurations, change the line<br>
`configs = [config.Config(mode=BMEPMode.BunV, init_file_path="ini/main.ini")]`<br>
in "main.py".
Inside "src/ini", we added different preset configurations:
- "config_F1_Manifold.ini" is used for **F1-Manifold**.
- "config_F1_Buneman_disjunctive.ini" is used for **F1-Buneman**.
- "config_F1_Buneman_disjunctive.ini" is used for non **disjunctive F1-Buneman**.
- "config_contractions.ini" is used for **contraction-based formulation with UBT-manifold condition**.
- "config_contractions_Manifold.ini" is used for **contraction-based formulation with UBT-manifold condition**.

## Structure of .ini configuration files in ini folder

- name: (string) Name of the folder for the results file
- n_min: (int) Minimum value of n of the instances to be considered
- n_max: (int) Maximum value of n of the instances to be considered
- manifold: (bool) Activates UBT-manifold condition iff True
- time_limit: (float) Maximum time allowed for computation for each instance
- mipgap: (float) Maximum allowed optimality gap
- buneman_violation: (bool) Imposes violation of Buneman' strong four-point condition iff True
- buneman_disjunctive: (bool) Imposes Buneman' strong four-point condition iff True
- buneman_custom_constraints: (bool) Loads fixed values of the y binary variables in Formulation 1 iff True
- circular_orders_custom_constraints: (bool) Loads preset circular order facet-defining inequalities on n - 1 leaves iff True
- circular_orders_custom_filepath: (string) Path to preset circular order facet-defining inequalitie
- solver_mode: (categorical) SolverMode.Barebone (disable preprocessing and internal cuts) / SolverMode.Turbo (aggressive preprocessing, high heuristic frequency, high optimality gap) / SolverMode.Default (default options)
- instance_path: (string/categorical) Path to instances
- mode: (categorical)
- order_first_row: (bool) Imposes non-decreasing order on the first row of tau matrix iff True
- entries_equal_to_two: (bool) imposes the number of entries equal to 2 in upper triangular (or lower triangular) matrix
- fast_branching: (bool) Increases branching priority for lower values of tau, i.e., imposes a higher branching priority for x_{i, j}^l w.r.t. to each x_{i, j}^{l'} with l < l'
- start: (int) initial number for the matrices that violate Buneman conditions when execution mode is Buneman Violation
- repeat: (int) Number of times "main.py" iterates, for each fixed j = entries_equal_to_two, ..., 2, trying to obtain infeasibility when execution mode is Buneman Violation

