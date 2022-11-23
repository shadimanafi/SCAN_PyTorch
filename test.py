import torch

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = inceptionv1(pretrained=True)
_ = model.to(device).eval()

_ = render.render_vis(model, "mixed4a:476", save_image=True, image_name="test.jpg", show_image=False,thresholds=(8,),)
