#!/bin/bash

module load PrgEnv-nvidia cudatoolkit python gcc/9.3.0
#note cudatoolkit is currently 11.4

#conda insists on bringing down its own cudatoolkit, so we'll specify the right version
conda create -n wrf_eke python=3.8 -y
source activate wrf_eke
mamba install -c rapidsai-nightly -c nvidia -c conda-forge cudatoolkit=11.5 cudf=22.04 cython dask dask-cuda dask-cudf distributed ipykernel ipywidgets automake make libtool pkg-config psutil setuptools -y

#first install ucx
cd $SCRATCH
if [ -d "ucx" ]; then rm -Rf ucx; fi
git clone https://github.com/openucx/ucx
cd ucx
git checkout v1.12.1
./autogen.sh
mkdir build
cd build
../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I$CUDA_HOME/include"
make -j 16 install

#and then install ucx-py from source
#cd $SCRATCH
#if [ -d "ucx-py" ]; then rm -Rf ucx-py; fi
#git clone https://github.com/rapidsai/ucx-py
#cd ucx-py
#python setup.py build
#python setup.py install
python -m pip install --no-cache-dir git+https://github.com/rapidsai/ucx-py
