work_path=$(dirname $0)
model_name='gridtosim'
now=$(date +%s)

PYTHONPATH=./ python -u $work_path/../run.py --exp-config $work_path/no_learning.yaml \
--run-type eval 
#\
#2>&1 | tee $work_path/train.$now.log.out