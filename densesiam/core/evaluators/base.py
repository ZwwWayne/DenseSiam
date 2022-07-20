# Copyright (c) Facebook, Inc. and its affiliates.
from collections import OrderedDict

from densesiam.utils import get_root_logger
from densesiam.utils.comm import is_main_process


class DatasetEvaluator:
    """Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over all samples
    in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by
    :meth:`process`), and produce evaluation results in the end (by
    :meth:`evaluate`).
    """

    def reset(self):
        """Preparation for a new round of evaluation.

        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """Process the pair of inputs and outputs. If they contain batches, the
        pairs can be consumed one-by-one using `zip`:

        .. code-block:: python
            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluatorsWrapper(DatasetEvaluator):
    """Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to all of its
    :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        logger = get_root_logger()
        logger.info('Reset Evaluators')
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        logger = get_root_logger()
        logger.info('Start evaluation after inference')
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert k not in results, \
                        ('Different evaluators produce '
                         f'results with the same key {k}')
                    results[k] = v
        return results
