work_path=$(dirname $0)
model_name='seq2seq'
now=$(date +%s)

PYTHONPATH=./ python -u $work_path/../run.py --exp-config $work_path/seq2seq.yaml \
--run-type train 
#\
#2>&1 | tee $work_path/train.$now.log.out