# Time Series Insight Toolkit

Analytics tool for time series data. Developed for csv records captured from 3D web applications.

## Installation

### Dependencies

The dependencies are stored in `environment.yml`, and are:
```
name: mat
channels:
  - defaults
dependencies:
  - jupyter
  - numpy
  - matplotlib
  - pandas
  - scipy
  - scikit-learn
  - ipympl
  - seaborn
  - voila
```

to install dependencis with conda:
```
conda env create -f environment.yml
```

### Test

To test script

```
python -m doctest timeSeriesInsightToolkit.py
```

To test script in verbose mode
```
python -m doctest -v timeSeriesInsightToolkit.py
```

## Usage

### Run Preprocessing

To run preprocessing, you can get a voila session started:
```
voila preprocess-sessions-collection-voila2.ipynb --Voila.ip='*' --port=XXXX --no-browser --debug
```
then you can access the ip adress with the specified port XXXX.

### Run gate

To run gate on 3D VR enviroments:
```
 python gate.py --path path/to/group/folder --write
```
to run gate on 3D VR panoramic views:
```
 python gate.py --path path/to/group/folder --panoramic --write
```
the `--write` argument is used to write the register that are not gated.
In addition, there are two extra arguments, the first --show to show the generated plots when the script is generated, and `--noqt` to disable qt.

### Make 3D path preview

To make png preview of 3D paths and put preview in same folder, pnd files have `-prev` suffix added
```
python make3dprev.py --path path/to/group/folder
```
To make png preview of 3D paths and put in different output folder
```
python make3dprev.py --path path/to/group/folder --opath path/to/group
```

### Gaussian Kernel density estimation (KDE)

To compute the 3d kde of the positions (pos), by defolt the kde is discretized on voxels with `width=0.4`, to chage this use `--width`
```
python generate-kde.py --path path/to/group/folder --opath path/to/group/
```
To compute the 2d projection kde of the positions (pos)
```
python generate-kde.py --path path/to/group/folder --opath path/to/group/ --proj2d
```
To compute the 3d kde of the directions (dir)
```
python generate-kde.py --path path/to/group/folder --opath path/to/group/ --dir
```
To compute the 3d kde of the directions (dir) in panoramic scenes
```
python generate-kde.py --path path/to/group/folder --opath path/to/group/ --dir --panoramic
```

### Agglomerative clustering

`pos-clustering-kde.ipynb` and `panoramic-clustering-kde.ipynb` are Jupyter notebooks for applying agglomerative clustering on records kde.

### K-means

`pos-k-means.ipynb` are Jupyter notebooks for applying k-means to the record varaibles.


### Gaussian mixture morel

`pos-gmm.ipynb` are Jupyter notebooks for applying k-means to the record varaibles.


Compute k-means on records. 


## Rest API (Experimental)

Simple Rest API implementation.

### Preprocessing

Show records:
```
http://<ipaddress>:<port>/<group>/<subgroup>/
```

Show record data:
```
http://<ipaddress>:<port>/<group>/<subgroup>/<record>
```

Plot record data:
```
http://<ipaddress>:<port>/<group>/<subgroup>/<record>/plot
```

### Run gate

Get all measures defined for each record.
```
http://<ipaddress>:<port>/<group>/<subgroup>/measures
```
Filter measures, at the moment only doration and variance are implemented.
```
http://<ipaddress>:<port>/<group>/<subgroup>/measures?key=variance
```
or
```
http://<ipaddress>:<port>/<group>/<subgroup>/measures?key=variance,duration
```


Generate scatter plot of two variables for which there is a defined function.
```
http://<ipaddress>:<port>/<group>/<subgroup>/scatter?var1=duration>35&var2=variance>0.4
```
Generate scatter plot of two variables for which there is a defined function, and show gate threshold
```
hhttp://<ipaddress>:<port>/<group>/<subgroup>/scatter?var1=duration>35&var2=variance>0.4
```
