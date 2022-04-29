from setuptools import setup

setup(
    name='wrf_eke_example',
    version='0.0.1',
    description='Weather Research and Forecasting Model Eddy Kinetic Energy example analysis',
    packages=['wrf_eke_example'],
    scripts=[
        'bin/wrf_eke_example_run.py',
        'bin/WRF_EKE_NERSC_Ex.py',
        'bin/launch-dask-cpu.sh',
        'bin/launch-dask-gpu.sh'
        ]
)
