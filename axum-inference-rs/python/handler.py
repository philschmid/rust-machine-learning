from transformers import pipeline


class Handler(object):
    def __init__(self, task, **kwargs):
        self.pipeline = pipeline(task)

    def __call__(self, inputs):
        return self.pipeline(inputs)
