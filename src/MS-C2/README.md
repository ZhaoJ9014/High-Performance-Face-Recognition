### There are 3 files should get from extraction:


- The novel gallery, base gallery and testing features.


- The 3 files are processed in child folder by following py files:

    - cmb.py: to combine two galleries.


    - ext.py: to extract the ids as "slave.txt".


    - trans.py: to align the order of slave to "master.txt" and generate "trans.txt"


    - testall.py: to get the final result from testing features, gallery features and trans.txt.


- Then copy all the final results to root folder, edit getLabel.py to modify your voting strategy.
