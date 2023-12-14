# import loader
import argparse
import math
from image_loader import image_process
from models import *
import numpy as np

if __name__ == "__main__":
    parser =  argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required = True, type = str)
    parser.add_argument("-s", "--size", default = 256, type = int)
    parser.add_argument("-l", "--latent_size", default = 100, type = str)
    parser.add_argument("-b", "--batch_size", default = 32, type = str)

    args = parser.parse_args()
    
    data = image_process(args.dir, (args.size, args.size))
    data = np.array(data)
    
    generator = define_generator(args.latent_size, args.size)
    discriminator = define_discriminator(args.size)

    generator.summary()
    discriminator.summary()

    model = WGAN_GP(generator, discriminator, latent_size = args.latent_size)

    model.compile(tf.keras.optimizers.RMSprop(learning_rate = 1e-4),
                  tf.keras.optimizers.RMSprop(learning_rate = 1e-4))
    model.fit(data, epochs = 10000, callbacks = [Monitor(10, __file__.replace("main.py", ""))], shuffle = False)

    
