# -*- coding: utf-8 -*-
from recbole.config import Config


class MLFlowLogger(object):
    def __init__(self, config: Config):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        try:
            self.mlflow_config = config.mlflow
        except AttributeError:
            self.mlflow_config = {}

        self.enabled = self.mlflow_config.get('enable', False)
        self.setup()

    def setup(self):
        if self.enabled:
            try:
                from mlflow.tracking import MlflowClient

            except ImportError:
                raise ImportError(
                    "To use the MLFlow Logger please install mlflow."
                    "Run `pip install mlflow` to install it."
                )

            self.mlflow_client = MlflowClient(self.mlflow_config['tracking_uri'])
            exp_name = self.mlflow_config['experiment_name']
            # get or create experiment
            try:
                self.exp_id = self.mlflow_client.create_experiment(exp_name)
            except:
                self.exp_id = self.mlflow_client.get_experiment_by_name(exp_name).experiment_id

            run = self.mlflow_client.create_run(experiment_id=self.exp_id, run_name=self.mlflow_config['run_name'])
            self.run_id = run.info.run_id

            for k, v in self.config.final_config_dict.items():
                self.mlflow_client.log_param(self.run_id, k, v)

    def log_model_params(self, model):
        if self.enabled:
            self.mlflow_client.log_param(self.run_id, "model_params", model.get_num_params())

    def finish_training(self, status):
        self.mlflow_client.set_terminated(self.run_id, status)

    def log_metrics(self, metrics, epoch, head="train", commit=True):
        if self.enabled:
            if head:
                metrics = self._add_head_to_metrics(metrics, head)

            for k, v in metrics.items():
                self.mlflow_client.log_metric(self.run_id, k, v, step=epoch)

    def log_eval_metrics(self, metrics, epoch, head="eval"):
        if self.enabled:
            metrics = self._add_head_to_metrics(metrics, head)
            for k, v in metrics.items():
                self.mlflow_client.log_metric(self.run_id, k, v, step=epoch)

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            k = k.replace('@', '_at_')
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}"] = v

        return head_metrics


def flatten_dict(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = prefix_dict(k, flatten_dict(v), sep='.')
            res.update(v)
        else:
            res[k] = v
    return res


def prefix_dict(p, d, sep='_'):
    return {f'{p}{sep}{k}': v for k, v in d.items()}
