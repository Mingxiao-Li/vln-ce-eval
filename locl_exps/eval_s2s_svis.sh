work_path=$(dirname $0)
model_name='seq2seq_small_vis'
now=$(date +%s)

PYTHONPATH=./ python -u $work_path/../run.py --exp-config $work_path/seq2seq_eval.yaml \
--run-type eval \
LOG_FILE $work_path/$model_name/train.log \
NUM_ENVIRONMENTS 1 \
TENSORBOARD_DIR data/tensorboard_dirs/$model_name \
EVAL_CKPT_PATH_DIR data/checkpoints/$model_name \
RESULTS_DIR data/checkpoints/$model_name/eval \
TASK_CONFIG.DATASET.DATA_PATH data/datasets/annt/{split}/{split}.json.gz \
IL.RECOLLECT_TRAINER.gt_file "data/datasets/annt/{split}/{split}_gt.json.gz " \
IL.DAGGER.lmdb_features_dir "data/trajectories_dirs/$model_name/tra.lmdb" \
MODEL.STATE_ENCODER.hidden_size 256 \
IL.DAGGER.update_size 3000 \
#\