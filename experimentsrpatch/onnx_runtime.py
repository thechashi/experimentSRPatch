import onnx
import torch
import utilities as ut

onnx_model = onnx.load("inference_models/rrdb.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("inference_models/rrdb.onnx")


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


img_path = "data/slices/13.npz"
input_batch = ut.npz_loader(img_path).unsqueeze(0)

input_ = torch.tensor(input_batch).int()
output_folder = "output_images"
file_name = img_path.split("/")[-1].split(".")[0]
ut.save_image(input_[0], output_folder, 30, 30, 4, output_file_name=file_name + "_input_x4")

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
ut.save_image(output[0], output_folder, 120, 120, 4, output_file_name=file_name + "_x4")
