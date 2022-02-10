from jsii import data_type
from tritonclient.utils import *
import tritonclient.http as httpclient
import time
import json
import numpy as np

model_name = "pipeline"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

text = "I like you. I love you"

triton_client = httpclient.InferenceServerClient(url=url, verbose=False)

model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

query = httpclient.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
model_score = httpclient.InferRequestedOutput(name="PREDICTION", binary_data=False)

query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
)

resp = json.loads(response.get_response()["outputs"][0]["data"][0])
print(resp)
