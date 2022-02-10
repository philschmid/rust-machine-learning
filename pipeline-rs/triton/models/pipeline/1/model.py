import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import os
import numpy as np
import json


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(args["model_repository"], args["model_version"]))

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            input_as_triton_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            decoded_input = input_as_triton_tensor.as_numpy()[0].decode("utf-8")

            # preprocess
            inputs = self.tokenizer(decoded_input, return_tensors="np")
            processed_input = [
                pb_utils.Tensor("input_ids", inputs["input_ids"]),
                pb_utils.Tensor("attention_mask", inputs["attention_mask"]),
            ]

            # CALL BERT Model
            infer_request = pb_utils.InferenceRequest(
                model_name="bert",
                requested_output_names=[
                    "probabilities",
                ],
                inputs=processed_input,
            )
            # Perform synchronous blocking inference request
            infer_response = infer_request.exec()

            # Make sure that the inference response doesn't have an error. If
            # it has an error and you can't proceed with your model execution
            # you can raise an exception.
            if infer_response.has_error():
                raise pb_utils.TritonModelException(infer_response.error().message())

            # Postprocess request
            scores = softmax(infer_response.output_tensors()[0].as_numpy()[0])
            res = {"label": scores.argmax().item(), "score": scores.max().item()}

            out_tensor_0 = pb_utils.Tensor("PREDICTION", np.array([json.dumps(res)], dtype=object))

            # Create InferenceResponse.
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
