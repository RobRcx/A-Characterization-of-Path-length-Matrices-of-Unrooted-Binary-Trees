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

## Structure of the repository

Folders:<br>
|---> data (problem instances used in Section 4 and Appendix A)<br>
|---> results (results presented in Section 4 and Appendix A)<br>
|---> src (source code files)

## Support code for Proposition 12

- "combinations/generate_combinations.py" is the main script for generating the combinations of path-length sequences.<br>
- "config.py" is a configuration script containing values for the parameters "n" (i.e, the number of leaves) and "solver", which depends on the MILP solver installed on the runtime platform (e.g., "cplex" for IBM CPLEX and "xpress" for FICO Xpress, according to pyomo documentation: https://pyomo.readthedocs.io/en/stable/).

## Code used in Section 4, Appendix A, and Appendix C

- "main.py" is the main script for using F1-Manifold and F1-Buneman (both with and without disjunctive constraints).<br>
- "ini" folder contains the different configurations described in Section 4. We added several preset configurations, but the one that is loaded by "main.py" is "ini/main.ini". In order to load different configurations, change the line<br>
`configs = [config.Config(mode=BMEPMode.BunV, init_file_path="ini/main.ini")]`<br>
to add different configurations.

