work_path=$(dirname $0)
model_name='seq2seq_small'
now=$(date +%s)

PYTHONPATH=./ python -u $work_path/../run.py --exp-config $work_path/seq2seq.yaml \
--run-type train \
TENSORBOARD_DIR data/tensorboard_dirs/$model_name \
CHECKPOINT_FOLDER data/checkpoints/$model_name \
EVAL_CKPT_PATH_DIR data/checkpoints/$model_name \
RESULTS_DIR data/checkpoints/seq2seq/$model_name/evals\
IL.DAGGER.lmdb_features_dir data/trajectories_dirs/$model_name/tra \
TASK_CONFIG.DATASET.DATA_PATH data/datasets/annt/{split}/{split}.json.gz \
IL.DAGGER.update_size 5000 \
#\
#2>&1 | tee $work_path/train.$now.log.out