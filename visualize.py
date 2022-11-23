import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

class filter1():
    def plot_filters_single_channel_big(t, title):
        # setting the rows and columns
        nrows = t.shape[0] * t.shape[2]
        ncols = t.shape[1] * t.shape[3]

        npimg = np.array(t.cpu().numpy(), np.float32)
        npimg = npimg.transpose((0, 2, 1, 3))
        npimg = npimg.ravel().reshape(nrows, ncols)

        npimg = npimg.T

        fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
        imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
        plt.title(title, fontsize=5)
        plt.show()

    def plot_filters_single_channel(t, title):
        # kernels depth * number of kernels
        nplots = t.shape[0] * t.shape[1]
        ncols = 12

        nrows = 1 + nplots // ncols
        # convert tensor to numpy image
        npimg = np.array(t.cpu().numpy(), np.float32)

        count = 0
        fig = plt.figure(figsize=(ncols, nrows))

        # looping through all the kernels in each channel
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                count += 1
                ax1 = fig.add_subplot(nrows, ncols, count)
                npimg = np.array(t[i, j].cpu().numpy(), np.float32)
                npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                ax1.imshow(npimg)
                ax1.set_title(str(i) + ',' + str(j))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
        plt.title(title, fontsize=5)
        plt.tight_layout()
        plt.show()

    def plot_filters_multi_channel(t, title):
        # get the number of kernals
        num_kernels = t.shape[0]

        # define number of columns for subplots
        num_cols = 12
        # rows = num of kernels
        num_rows = num_kernels

        # set the figure size
        fig = plt.figure(figsize=(num_cols, num_rows))

        # looping through all the kernels
        for i in range(t.shape[0]):
            ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

            # for each kernel, we convert the tensor to numpy
            npimg = np.array(t[i].numpy(), np.float32)
            # standardize the numpy image
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            npimg = npimg.transpose((1, 2, 0))
            ax1.imshow(npimg)
            ax1.axis('off')
            ax1.set_title(str(i))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        plt.savefig('myimage.png', dpi=100)
        plt.tight_layout()
        plt.title(title, fontsize=5)
        plt.show()

    def plot_weights(self,model, layer_num, single_channel=True, collated=False, title=None):
        # extracting the model features at the particular layer number
        layer = model[layer_num]

        # checking whether the layer is convolution layer or not
        if isinstance(layer, nn.Conv2d):
            # getting the weight tensor data
            weight_tensor = model[layer_num].weight.data

            if single_channel:
                if collated:
                    self.plot_filters_single_channel_big(weight_tensor, title)
                else:
                    self.plot_filters_single_channel(weight_tensor, title)

            else:
                if weight_tensor.shape[1] == 3:
                    self.plot_filters_multi_channel(weight_tensor, title)
                else:
                    print("Can only plot weights with three channels with single channel = False")

        else:
            print("Can only visualize layers which are convolutional")


class filter2():
    def __init__(self,model):
        self.model_weights = []  # we will save the conv layer weights in this list
        self.conv_layers = []  # we will save the 49 conv layers in this list
        # get all the model children as list
        self.model_children = list(model.net.encoder.children())

    def access_layers(self):
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective weights to the list
        for i in range(len(self.model_children)):
            if type(self.model_children[i]) == nn.Conv2d:
                counter += 1
                self.model_weights.append(self.model_children[i].weight.cpu())
                self.conv_layers.append(self.model_children[i].cpu())
            elif type(self.model_children[i]) == nn.Sequential:
                for j in range(len(self.model_children[i])):
                    for child in self.model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            self.model_weights.append(child.weight)
                            self.conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")

    def visualize_filter(self):
        # visualize the first conv layer filters
        plt.figure(figsize=(20, 17))
        for i, filter in enumerate(self.model_weights[0]):
            plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
            plt.axis('off')
            # plt.savefig('../outputs/filter.png')
        plt.show()

    def visualize_feature(self,img):
        # plt.imshow(np.einsum('zxy->xyz',img.cpu().squeeze(0).detach().numpy()))
        # plt.title("original image")
        # plt.show()
        # pass the image through all the layers
        results = [self.conv_layers[0](img)]
        for i in range(1, len(self.conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(self.conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results

        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())
            for i, filter in enumerate(layer_viz):
                if i == 64:  # we will visualize only 8x8 blocks from each layer
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter, cmap='gray')
                plt.axis("off")
            print(f"Saving layer {num_layer} feature maps...")
            # plt.savefig(f"../outputs/layer_{num_layer}.png")
            plt.title(f"Saving layer {num_layer} feature maps")
            plt.show()
            # plt.close()
