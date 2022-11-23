import torch

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1


def sample_feature(model,path_root,starting_layer,number_of_layers,channels_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    for layer in range(starting_layer,starting_layer+number_of_layers):
        for channel in range(0,channels_num[layer]):
            s=str(layer)+":"+str(channel)
            print(s)
            _ = render.render_vis(model, s, save_image=True, image_name=path_root+"/test"+s+".jpg", show_image=False,)

def feature_transform(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    obj = objectives.channel("0", 0)
    all_transforms = [
        transform.pad(16),
        transform.jitter(8),
        transform.random_scale([n / 100. for n in range(80, 120)]),
        transform.random_rotate(list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))),
        transform.jitter(2),
    ]

    _ = render.render_vis(model, obj,transforms=all_transforms, save_image=True, image_name="test.jpg", show_image=False,)
