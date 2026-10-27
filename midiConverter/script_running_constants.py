# Training command
RUN_CNN_DEFAULT = "python -m src.baseline.models.dl_classifiers.trainer.single_label_trainer --model-type cnn"

# Evaluation command
EVAL_CNN_DEFAULT = "python -m src.baseline.evaluation.model_evaluator_runner --model-type cnn --save-results"
