# cvrVSrcvr
Tools to compare CVR and rCVR mapping methods

You must download the data from openneuro to run this script. You need ds004604 and ds005418. The expected location is /data/ds004604 and /data/ds005418.
You also need to install the following python packages: numpry, os, itertools, nilearn, pathlib, pybids, pandas, xml, json and warnings
Make also sure you have the following files installed (see fsl installation to get then):
'/opt/fsl/data/atlases/HarvardOxford-Cortical.xml'
'/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-prob-1mm.nii.gz'

To run the script, navitage to cvrVSrcvr/cvrVSrcvr and execute `python cvrVSrcvr.py'
