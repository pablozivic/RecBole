# -*- coding: utf-8 -*-


class MLFlowLogger(object):
    def __init__(self, config):
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

    def log_metrics(self, metrics, head="train", commit=True):
        if self.enabled:
            if head:
                metrics = self._add_head_to_metrics(metrics, head)

            self.mlflow_client.log_metrics(metrics)

    def log_eval_metrics(self, metrics, head="eval"):
        if self.enabled:
            metrics = self._add_head_to_metrics(metrics, head)
            self.mlflow_client.log_metrics(metrics)

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}"] = v

        return head_metrics
