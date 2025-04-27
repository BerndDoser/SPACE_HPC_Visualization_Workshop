import wandb

from spherinator.models import export_onnx

project_name = "gaia"
model_name = "model-c7fudnr9:v0"

api = wandb.Api()
artifact = api.artifact(f"ain-space/{project_name}/{model_name}", type="model")
artifact.download()

export_onnx(
    ckpt_file=f"artifacts/{model_name}/model.ckpt",
    model_file="configs/spherinator/gaia.yaml",
    export_path="data/gaia/models/full_trained",
    input_shape=(2, 1, 343),
    latent_shape=(2, 3),
)
