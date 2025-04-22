import wandb

from spherinator.models import export_onnx

api = wandb.Api()
artifact = api.artifact("ain-space/illustris/model-8n67t45m:v0", type="model")
artifact.download()

export_onnx(
    ckpt_file="artifacts/model-8n67t45m:v0/model.ckpt",
    model_file="configs/spherinator/illustris.yaml",
    export_path="data/illustris/models",
    input_shape=(2, 3, 128, 128),
    latent_shape=(2, 3),
)
