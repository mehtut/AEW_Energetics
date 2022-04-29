## WRF (Weather Research and Forecasting Model) EKE (Eddy Kinetic Energy) Example code to run at NERSC

This example processes a gridded dataset that was created using the 
Weather Research and Forecasting (WRF) model,
containing vectors (U and V components) at different pressure levels.  This is a 10 year dataset
with yearly files, one file per vector component per year, or 20 files total, approximately 150GB.

The example task is to compute the average Eddy Kinetic Energy (EKE) for the full timespan of data available
using a bounding box defined within the gridded data and then produce a figure plot of the result.
The data is also filtered to remove noise.

### Installation/Setup

#### 1. Clone this repo

```bash
git clone https://github.com/mlhenderson/AEW_Energetics
```

#### 2. Create a conda environment with dependencies

```bash
module load python
# using a GPU environment (Perlmutter)
conda create env -n wrf_eke -f env/gpu_dependencies.yml
# OR using a CPU only environment (Cori)
conda create env -n wrf_eke -f env/cpu_dependencies.yml
```

##### Activating the conda environment
```bash
# run the module load once per login shell
# module load python
source activate wrf_eke
```

#### 3. Install example code

```bash
pip install .
```

#### 4. Create a Jupyter kernel to run Notebooks

```bash
# make sure to have the conda environment activated
python -m ipykernel install --user --name wrf_eke --display-name wrf_eke
```

#### 5. How to run

##### From an salloc or sbatch job

###### CPU
```bash
module load python
source activate wrf_eke
launch-dask-cpu.sh
```

###### GPU
```bash
module load python
source activate wrf_eke
launch-dask-gpu.sh
```

##### Command line tool

```bash
wrf_eke_example.py --help

usage: wrf_eke_example_run.py [-h] --data_path DATA_PATH
                              [--scheduler_file SCHEDULER_FILE]
                              [--scenario SCENARIO] [--start_year START_YEAR]
                              [--end_year END_YEAR] [--dataset DATASET]
                              [--chunksize_lev CHUNKSIZE_LEV]
                              [--chunksize_y CHUNKSIZE_Y]
                              [--chunksize_x CHUNKSIZE_X] [--backend BACKEND]

Calculate Average Eddy Kinetic Energy for a set of vectors with LEV

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the input vector data files
  --scheduler_file SCHEDULER_FILE
                        Dask scheduler file to connect to an existing cluster
  --scenario SCENARIO   default is Historical
  --start_year START_YEAR
                        First year to process
  --end_year END_YEAR   Last year to process
  --dataset DATASET     Name of the WRF data file for coordinates
  --chunksize_lev CHUNKSIZE_LEV
                        Number of LEV dimension per chunk
  --chunksize_y CHUNKSIZE_Y
                        Number of Y (LAT) dimension per chunk
  --chunksize_x CHUNKSIZE_X
                        Number of X (LON) dimension per chunk
  --backend BACKEND     Select a backend to run the example, "serial", "dask",
                        "dask_cuda"

```

##### Minimal arguments to run

```bash
wrf_eke_exmaple_run.py --data_path=<absolute payth to data>
```


#### Data

The files processed in the example are gridded data based on the Weather Research and Forecasting (WRF) model at NCAR:
- https://ncar.ucar.edu/what-we-offer/models/weather-research-and-forecasting-model-wrf
- https://www2.mmm.ucar.edu/wrf/users/

For the example data, there are 10 years of data, with two data files per year, one file for each component vector U and V.

#### Notebooks

There is a Jupyter Notebook "EKE_average.ipynb" that captures analysis steps from the original script, modified for dask distributed and Jupyter.

### Contents

- wrf_eke_example/
    - python code
- env/
    - files needed to create a working environment
- notebooks/
    - Jupyter Notebooks capturing the example analysis with some visualizations
