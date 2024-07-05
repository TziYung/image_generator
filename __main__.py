# import loader
import argparse
import math
from image_loader import *
from models import *

if __name__ == "__main__":
    parser =  argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required = True, type = str)
    parser.add_argument("-s", "--size", default = 256, type = int)
    parser.add_argument("-l", "--latent_size", default = 100, type = int)
    parser.add_argument("-b", "--batch_size", default = 32, type = int)

    args = parser.parse_args()
    
    data = ImageLoader(args.dir, (args.size, args.size), args.batch_size)
    
    generator = define_generator(args.latent_size, args.size)
    discriminator = define_discriminator(args.size)

    generator.summary()
    discriminator.summary()

    model = WGAN_GP(generator, discriminator, latent_size = args.latent_size)

    model.compile(tf.keras.optimizers.RMSprop(learning_rate = 1e-4),
                  tf.keras.optimizers.RMSprop(learning_rate = 1e-4))
    model.fit(data, epochs = 10000, batch_size = args.batch_size, callbacks = [Monitor(10, __file__.replace("__main__.py", ""))], shuffle = False)

    
