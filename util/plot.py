import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class gan_plotter():

    def __init__(self, opt):

        self.model_name = opt.name

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        self.graph = os.path.join(opt.checkpoints_dir, opt.name, 'graph.png')

    def plot_log(self): 
        data = []
        with open(self.log_name, "r") as file:
            for line in file:
                if (line[0] == "=" or len(line) < 26): 
                    continue

                # Splitting each line by spaces
                parts = line.replace(":",",").replace(" ",",").split(",")
                
                # Extracting relevant information
                epoch = int(parts[2])
                G_GAN = float(parts[17])
                G_L1 = float(parts[20])
                D_real = float(parts[23])
                D_fake = float(parts[26])
                val_G_GAN = float(parts[29])
                val_G_L1 = float(parts[32])
                SSIM = float(parts[35])

                # fix for missing line break
                if (parts[38].count("=") > 0):
                    PSNR = float(parts[38].split("=")[0])
                else:
                    PSNR = float(parts[38])

                # Appending to data
                data.append([epoch, G_GAN, G_L1, D_real, D_fake, val_G_GAN, val_G_L1, SSIM, PSNR])
        
        data = np.array(data)

        epochs = np.unique(data[:,0])[:-1] #We don't want the most recent epoch, as it may be unreliable

        if (len(epochs) == 0):
            return

        # Preallocate the means array
        means = np.zeros((len(epochs), 9))

        # Iterate over epochs and compute the means 
        for i, value in enumerate(epochs):
            mask = data[:, 0] == value
            subset = data[mask]

            means[i, 0] = value
            means[i, 1:] = np.mean(subset[:, 1:], axis=0)
        

        plt.style.use('default')
        plt.rcParams.update({'font.size': 16})

        fig, axs = plt.subplots(1, 4, figsize=(24, 4))

        self.plot_graph(axs, 0, "G Train", epochs, means[:, 1], line_color = "steelblue", plot_title = "G and D loss")
        self.plot_graph(axs, 0, "G Val", epochs, means[:, 5], line_color = "goldenrod")
        self.plot_graph(axs, 0, "D real", epochs, means[:, 3], line_color = "darkseagreen")
        self.plot_graph(axs, 0, "D fake", epochs, means[:, 4], line_color = "palevioletred")
        
        self.plot_graph(axs, 1, "Train", epochs, means[:, 2], line_color = "steelblue", plot_title = "L1 loss")
        self.plot_graph(axs, 1, "Val", epochs, means[:, 6], line_color = "goldenrod")
        
        self.plot_graph(axs, 2, "Val", epochs, means[:, 7], line_color = "goldenrod", plot_title = "SSIM")
        
        self.plot_graph(axs, 3, "Val", epochs, means[:, 8], line_color = "goldenrod", plot_title = "PSNR")
        
        plt.tight_layout()

        plt.tight_layout()
        plt.savefig(f"{self.graph}") 
        plt.close()


    def plot_graph(self, ax, ax_index, plot_label, x_vals, y_vals, line_color = "blue", plot_title = None):
        if (plot_title):
            ax[ax_index].set_title(plot_title)
        ax[ax_index].grid(visible=True)
        ax[ax_index].plot(x_vals, y_vals, label=plot_label, linewidth=1, color=line_color)
        ax[ax_index].set_xlabel("Epoch")
        ax[ax_index].set_ylabel("Loss")
        ax[ax_index].legend()
        ax[ax_index].spines['top'].set_visible(False)  # Hide the top spine for each subplot
        ax[ax_index].spines['right'].set_visible(False)  # Hide the top spine for each subplot
        ax[ax_index].yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, zorder=3)
        ax[ax_index].set_facecolor('#f0f0f0')


        # TODO Manual limits should be passed from... somewhere eventually

        # ax[ax_index].set_xlim(0, 80)  # Set the limit for the x-axis
        # ax[ax_index].xaxis.set_major_locator(MultipleLocator(10))

        # if (ax_index == 0):
        #     ax[ax_index].set_ylim(0, 6)  # Manually set the limit for the x-axis

        # if (ax_index == 1):
        #     ax[ax_index].set_ylim(0, 0.1)  # Set the limit for the x-axis

        # if (ax_index == 2):
        #     ax[ax_index].set_ylim(0.6, 0.78)  # Set the limit for the x-axis

        # if (ax_index == 3):
        #     ax[ax_index].set_ylim(19, 23)  # Set the limit for the x-axis              

    def rolling_avg(self, a,n): 
        # TODO do I want rolling avg?
        assert n%2==1
        b = a*0.0
        for i in range(len(a)) :
            b[i]=a[max(i-n//2,0):min(i+n//2+1,len(a))].mean()
        return b
