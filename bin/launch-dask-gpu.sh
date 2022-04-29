#!/bin/bash                                                                                   

echo "Starting scheduler..."

scheduler_file='scheduler_file.json'
rm -f $scheduler_file

#start scheduler                                                                              
UCX_MAX_RNDV_RAILS=1 \
UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda \
DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True \
UCX_MEMTYPE_CACHE=n \
UCX_TCP_MAX_CONN_RETRIES=255 \
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
dask-scheduler \
    --protocol ucx \
    --interface hsn0 \
    --scheduler-file $scheduler_file &
dask_pid=$!


# Wait for the scheduler to start                                                             
sleep 5
until [ -f $scheduler_file ]
do
     sleep 5
done


UCX_TCP_MAX_CONN_RETRIES=255 \
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
UCX_MAX_RNDV_RAILS=1 \
UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda \
UCX_MEMTYPE_CACHE=n \
srun --gpus-per-task 4 dask-cuda-worker \
    --device-memory-limit 38GB \
    --rmm-pool-size 38GB \
    --local-directory /tmp \
    --interface hsn0 \
    --scheduler-file $scheduler_file


echo "Killing scheduler"
kill -9 $dask_pid

