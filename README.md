Kit for Agnostic Radiance Processing  (KARP)

Intended to have multiple things like plotting and whatnot. Starting out with just code to take Eumetsat's PC scores and create reconstructed radiances. Will add MTG-IRS, but starting out use IASI PC 2.11 product to start.

Codes:
- getIasiEv.sh  - downloads IASI EV files from my ftp (too cheap to pay for git-lfs)
- create_dumps.py - cats together little files from eumetsat into 6 hour windows.
- karrpp.py     - kit for agnostic reconstructed radiance pre-processing
- plotMap.py    - script to plot either BUFR or H5/NC4/IODA output
- plotMatrix.py - script to plot various matrices associated with reconstructed radiances, along with instrument error covariance matrix.

Configuration Files:
- channe_subset.cfg - setup for channel subset and band scaling for BUFR  

