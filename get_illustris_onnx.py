import wandb

from spherinator.models import export_onnx

model_name = "model-56k0f026:v0"

api = wandb.Api()
artifact = api.artifact(f"ain-space/illustris/{model_name}", type="model")
artifact.download()

export_onnx(
    ckpt_file=f"artifacts/{model_name}/model.ckpt",
    model_file="configs/spherinator/illustris.yaml",
    export_path="data/illustris/models/full_trained",
    input_shape=(2, 3, 128, 128),
    latent_shape=(2, 3),
)
