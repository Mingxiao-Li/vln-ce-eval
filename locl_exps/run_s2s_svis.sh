work_path=$(dirname $0)
model_name='seq2seq_small_vis'
now=$(date +%s)

PYTHONPATH=./ python -u $work_path/../run.py --exp-config $work_path/seq2seq.yaml \
--run-type train \
LOG_FILE $work_path/$model_name/train.log \
TENSORBOARD_DIR data/tensorboard_dirs/$model_name \
CHECKPOINT_FOLDER data/checkpoints/$model_name \
RESULTS_DIR data/checkpoints/seq2seq/$model_name \
IL.DAGGER.lmdb_features_dir "data/trajectories_dirs/$model_name/tra.lmdb" \
TASK_CONFIG.DATASET.DATA_PATH data/datasets/annt/{split}/{split}.json.gz \
MODEL.STATE_ENCODER.hidden_size 256 \
IL.DAGGER.update_size 5000 \
#\
#2>&1 | tee $work_path/train.$now.log.out