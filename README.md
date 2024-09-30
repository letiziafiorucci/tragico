TRAGICO (TRends Analysis Guided Interfaces COllection) is a collection of functions for the extraction and analysis of experimental parameters from 1D and pseudo-2D NMR spectra acquired on Bruker instruments, 
designed to streamline the process of identifying trends in NMR data and facilitate various analytical tasks.

Key features include:

- Versatility: These functions are highly adaptable and can be integrated with external tools to accommodate a wide range of analytical needs.
- Customizability: Easily modify the functions to suit specific analysis requirements, ensuring flexibility and precision.
- User-Friendly: Clear examples and step-by-step instructions guide users through the application of the functions, making the process accessible to users of all levels.

Some routines included in the scripts are derived from the KLASSEZ package for NMR data processing and analysis (available at https://github.com/MetallerTM/klassez).

The code folder contains three python scripts: main4test.py, with example main codes, f\_fit.py, 
containing the functions for spectra modeling and integration, and f\_functions.py, a collection of all-purpose functions used by the analysis tools and in the input files generation. 
Additionally, a complete documentation and the results obtained from example codes are saved in dedicated folders.

The functions require python 3.12. The additional dependencies and their versions are: numpy (version 2.0.2), 
matplotlib (version 3.9.2), lmfit (version 1.3.2) and nmrglue (version 0.10). 

One way to set up the proper environment to run the codes is through the use of miniconda, available at https://docs.anaconda.com/miniconda/.
