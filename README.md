# Code for A CHARACTERIZATION OF PATH-LENGTH MATRICES OF UNROOTED BINARY TREES
The code in this repository is written in Python 3. The library dependencies of the code are contained in the Conda environment file "solve310.yml".
You can manually install all the dependencies therein, or you can optionally import this file as a Conda environment with the command:
<br>
conda env create -f solve310.yml
<br>
from the Anaconda prompt. The optional Anaconda installation can be downloaded at https://www.anaconda.com/download. 
 
<br><br>
The code uses DOCplex API for MILP. Academics and students can verify their access to a free version of CPLEX at the following link: https://community.ibm.com/community/user/ai-datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students?CommunityKey=ab7de0fd-6f43-47a9-8261-33578a231bb7&tab=

<br><br>



# Support code for Proposition 12

"combinations/generate_combinations.py" is the main script for generating the combinations of path-length sequences.
<br>
"config.py" is a configuration script containing values for the parameters "n" (i.e, the number of leaves) and "solver", which depends on the MILP solver installed on the runtime platform (e.g., "cplex" for IBM CPLEX and "xpress" for FICO Xpress, according to pyomo documentation: https://pyomo.readthedocs.io/en/stable/).

# Code used in Section 4, Appendix A, and Appendix C

"main.py" is the main script for using F1-Manifold and F1-Buneman (both with and without disjunctive constraints).
