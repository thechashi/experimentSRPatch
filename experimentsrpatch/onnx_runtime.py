import onnx
import torch
import utilities as ut

onnx_model = onnx.load("edsr.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("edsr.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

img_path = "data/test2.jpg"
input_batch = ut.load_image(img_path).unsqueeze(0)
print(input_batch.shape)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_batch)}
execution_time = ut.timer()
ort_outs = ort_session.run(None, ort_inputs)
execution_time = execution_time.toc()
print(execution_time)
output = ort_outs[0]

print(output)
print(output.shape)

output = torch.tensor(output).int()
output_folder = "output_images"
file_name = img_path.split("/")[-1].split(".")[0]
ut.save_image(output[0], output_folder, 100, 100, 4, output_file_name=file_name + "_x4")