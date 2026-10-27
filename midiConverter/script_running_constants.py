# Training commands
RUN_CNN_CQT_DEFAULT = "python -m src.baseline.models.dl_classifiers.trainer.single_label_trainer --model-type cnn --feature-type cqt"
RUN_CNN_MEL_DEFAULT = "python -m src.baseline.models.dl_classifiers.trainer.single_label_trainer --model-type cnn --feature-type mel"

# Evaluation commands
EVAL_CNN_CQT_DEFAULT = "python -m src.baseline.evaluation.model_evaluator_runner --model-type cnn --feature-type cqt --save-results"
EVAL_CNN_MEL_DEFAULT = "python -m src.baseline.evaluation.model_evaluator_runner --model-type cnn --feature-type mel --save-results"
