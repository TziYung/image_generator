from .wgan_gp import *
from matplotlib import pyplot as plt
import os
class Monitor(tf.keras.callbacks.Callback):
    def __init__(self, number, path):
        self.number = number
        self.path = path
        self.image_dir = os.path.join(self.path, "generated_images")
        self.weight_dir = os.path.join(self.path, "model_weighs")
        if os.path.isdir(self.image_dir) == False:
            os.makedir(self.image_dir)
        if os.path.isdir(self.weight_dir) == False:
            os.makedir(self.weight_dir)
    
    def on_epoch_end(self, epoch, logs = None):
        # To avoid log 0 and make it start with 1
        epoch = epoch + 1
        if (epoch % (10 ** int((math.log(epoch) / math.log(10))))) != 0:
            return None
        latent = tf.random.normal((self.number, self.model.latent_size))
        g_image = (self.model.generator(latent)  + 1) / 2
        plot = plt.figure(dpi = 600)
        row = int(self.number / 5 + min(1, self.number % 5))
        for n in range(self.number):
            img_holder = plot.add_subplot(row ,min(self.number, 5),n + 1)
            img_holder.set_xticks([])
            img_holder.set_yticks([])
            img_holder.imshow(g_image[n])
        plot.savefig(os.path.join(self.image_dir, f"epoch_{epoch}.png"))
        self.model.generator.save(os.path.join(self.weight_dir, f"generator_{epoch}.keras"))
        self.model.discriminator.save(os.path.join(self.weight_dir, f"discriminator_{epoch}.keras"))
        plt.close()

