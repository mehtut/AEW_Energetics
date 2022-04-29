#!/bin/bash

# get the number of hardware cores, multiply number of cores per socket by number of sockets
hw_cores=`lscpu | egrep 'Socket\(s\)|socket' | cut -d ":" -f 2 | sed -e 's/^[[:space:]]*//'| paste -d"*" - - | bc`

total_mem=`cat /proc/meminfo | grep MemTotal | cut -d ":" -f 2 | sed -e 's/^[[:space:]]*//g' | cut -d " " -f 1`

gigabyte="1048576"
total_mem=$(($total_mem/$gigabyte))

SHARED_NODE=0
while [[ $# -gt 0 ]]
do
  key="$1"

  case $key in
    -N|--nodes)
    NODES="$2"
    shift # past argument
    shift # past value
    ;;
    -n|--num_procs)
    NUMPROCS="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--num_threads)
    NUMTHREADS="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--shared)
    SHARED_NODE=1
    shift # past argument
    ;;
    -w|--working_dir)
    WORKDIR="$2"
    shift # past argument
    shift # past value
    ;;
  esac
done

num_nodes=$NODES

if [ "$num_nodes" = "" ]
then
  num_nodes=$SLURM_JOB_NUM_NODES
fi

num_procs=$NUMPROCS

if [ "$num_procs" = "" ]
then
  num_procs=$hw_cores
fi

num_threads=$NUMTHREADS

if [ "$num_threads" = "" ]
then
  num_threads=1
fi

working_dir=$WORKDIR

if [ "$working_dir" = "" ]
then
    now=`date -I'date'`
    working_dir="$SCRATCH/dask_clusters/${SLURM_JOB_ID}/${now}"
    mkdir -p ${working_dir}
fi


# Dask

start_dask_cluster()
{
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
  export OMP_NUM_THREADS=1
  export HDF5_USE_FILE_LOCKING=FALSE

  scheduler_file=${working_dir}/dask_scheduler.json

  # Get the IP address of our head node
  headIP=$(ip addr show ipogif0 | grep '10\.' | awk '{print $2}' | awk -F'/' '{print $1}')

  rm -f scheduler_file.json

  echo "Launching dask scheduler"
  DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
  DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
  dask-scheduler --host "$headIP" --scheduler-file $scheduler_file &

  while [ ! -f $scheduler_file ]
  do
      sleep 2
  done

  cp ${working_dir}/dask_scheduler.json scheduler_file.json

  STOP=$num_procs
  if [ $num_procs -ge $hw_cores ]
  then
      STOP=$(($num_procs - 2))
      echo "Maxing out cores on scheduler node, only launching $STOP workers on scheduler node"
  fi

  MEM=`echo "$total_mem/$STOP" | jq -nf /dev/stdin`

  echo "Launching $STOP dask workers on scheduler node with ${num_threads} threads and ${MEM}G memory per worker"
  COUNTER=0
  while [ $COUNTER -le $(($STOP - 1)) ]
  do
      DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
      DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
      dask-worker --local-directory ${working_dir} --scheduler-file ${working_dir}/dask_scheduler.json --nthreads $num_threads --memory-limit "${MEM}G" --reconnect &
      let COUNTER=COUNTER+1
  done

  MEM=`echo "$total_mem/$num_procs" | jq -nf /dev/stdin`

  # launch the rest of the workers, excluding the scheduler node
  if [ $SHARED_NODE = "0" ]
  then
      echo "Launching $num_procs dask workers with ${num_threads} and ${MEM}G memory per worker for $(($num_nodes - 1)) nodes"
      DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
      DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
      srun -N $(($num_nodes - 1)) --ntasks-per-node=$num_procs -x `hostname` dask-worker --local-directory ${working_dir} --scheduler-file ${working_dir}/dask_scheduler.json --nthreads $num_threads --memory-limit "${MEM}G" --reconnect
  fi
}

### Main

while [ ! -d ${working_dir} ]
do
    sleep 2
done

start_dask_cluster
