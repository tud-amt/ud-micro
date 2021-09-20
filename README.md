# UD Microstructural characterisation 

Method for the microstructural characterisation of unidirectional composite tapes (Katuin et al.)
This code by Nico Katuin presents a novel approach to identify  microstructural features both along the tape thickness and through the thickness. Voronoi tesselation based evaluation of the fibre volume content on cross sectional micrographs, with consideration of the matrix boundary, is performed. The method is shown to be robust and is suitable to be automated. It has the potential to discriminate specific microstructural features and to relate them to processing behaviour removing the need for manufacturing trials.
The original code is based on the master thesis of Nico Katuin at Delft University of Technology (https://repository.tudelft.nl/islandora/object/uuid%3A071a2a22-fa4d-423b-b049-6f2e70dbfc19) and has been further developed for this publication. 

# Use of code 
The MicrostructuralCharaterisation_v5.py code was originally written to compare  micrographs from three UD tapes from different suppliers. The fileloc variable can be used to specify which UD tape is analysed. In the original work of Katuin a powerful stitching functionality of a high resolution optical/confocal microscope was used. The obtained wide micrograph of a UD tape section was split into several width segments. This to increase the ease of manually defining the tape boundary. The amount of micrograph segments is not limited for the algorithm. Although, the width of the segments should be equal. 

The function blocks and function code are provided with comments in the .py file and should be self explanatory. 
