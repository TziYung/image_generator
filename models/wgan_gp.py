import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import math

class WGAN_GP(tf.keras.Model):
    def __init__(self, generator, discriminator, gp_weight = 10.0, latent_size = 10, noise_ratio = 0.1):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = gp_weight
        self.latent_size = latent_size
        self.noise_ratio = noise_ratio
        
        # Generator loss tracker
        self.g_loss_tracker = tf.keras.metrics.Mean(name = "Generator Loss")

        # Discriminator loss tracker
        self.d_loss_tracker = tf.keras.metrics.Mean(name = "Discriminator Loss")

        # Gradient Penalty tracker
        self.gp_tracker = tf.keras.metrics.Mean(name = "Gradient Penalty")

        # Total Discriminator Loss = Sum of gp and discriminator loss
        self.total_d_loss_tracker = tf.metrics.Mean(name = "Total Discriminator loss")

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.gp_tracker, self.total_d_loss_tracker]

    @tf.function
    def train_step(self, r_image):
        self.update_discriminator(r_image)
        self.update_generator(r_image)

        return {tracker.name: tracker.result() for tracker in self.metrics}

    def compile(self, g_opt, d_opt):
        super().compile()
        # Set the optimizer of generator
        self.g_opt = g_opt
        # Set the optimizer of discriminator
        self.d_opt = d_opt

    def update_generator(self, r_image):
        with tf.GradientTape() as tape:
            loss =self.get_generator_loss(tf.shape(r_image)[0])
        trainable_vars = self.generator.trainable_variables
        gradient = tape.gradient(loss, trainable_vars)
        self.g_opt.apply_gradients(zip(gradient, trainable_vars))

    def update_discriminator(self, r_image):
        latent = tf.random.normal(shape = (tf.shape(r_image)[0], self.latent_size))
        g_image = self.generator(latent)
        with tf.GradientTape() as tape:
            loss = self.get_discriminator_loss(r_image, g_image)
        trainable_vars = self.discriminator.trainable_variables
        gradient = tape.gradient(loss, trainable_vars)
        self.d_opt.apply_gradients(zip(gradient, trainable_vars))
    

    def get_discriminator_loss(self, r_image, g_image):
        # Adding noise to r_image
        r_image = r_image * (1.0 - self.noise_ratio) 
        r_image = r_image + tf.random.uniform(tf.shape(r_image), -1.0, 1.0) * self.noise_ratio
        
        r_score = self.discriminator(r_image)
        g_score = self.discriminator(g_image)
        gp = self.get_gp(r_image, g_image)
         
        # The goal is to make g_score and gp as small as possible and r_score as large as possible
        # The traget function would be r_score - g_score - gp -> the performance of discriminator (larger the better)
        # To transfer it to gradient descent applicable, it is multiplied by -1 so that:
        # -(r_score - g_score - gp) = -r_score + g_score + gp 
        # And we would be trying to minimize the above function 

        critic = -tf.reduce_mean(r_score - g_score)
        total_loss = critic + tf.reduce_mean(gp) * self.gp_weight 
        # Update trackers
        self.d_loss_tracker.update_state(critic)
        self.total_d_loss_tracker.update_state(total_loss)
        self.gp_tracker.update_state(gp)

        return total_loss
    def get_generator_loss(self, batch_size):
        latent = tf.random.normal((batch_size, self.latent_size))
        g_image = self.generator(latent)
        g_score = self.discriminator(g_image)
        # The goal of generator is to make g_score as large as possible.
        # The target function would be: g_score (larger the better)
        # To transfer it to gradient descent applicable function, it is multiplied -1:
        # - g_score
        # And we would try to minize -g_score 
        critic = -tf.reduce_mean(g_score)
        self.g_loss_tracker.update_state(critic)
        return critic
    
    def get_gp(self, r_image, g_image):
        """
        WGAN use wasserstein distance to measure the distance between real and generated image.
        Besides the loss calculated, it also needed the discriminator to be 1-lipschitz function.
        It is important because that if it is not, discriminator could just scaled up the output(ex: (1, 5), (100, 500)).
        Which doesn't help model classify real or generated image (similar to * 100 on the value when using mae, the value is larger but it doesn't really help training.)

        The paper of WGAN use the method of weight clipping, by forcing the weight not larger
        than the threshold(1), but it could cause the problem of model not converge. 
        The paper of WGAN-GP improved WGAN by replacing weight clippig with gradient penalty to WGAN.
        The goal of adding gradient penalty is to apply a soft constraint that rather than
        making discriminator 1-lipschitz function, it add gradient penalty so that the gradient
        of all the point between r_image and g_image and the prediction of discriminator of
        it are minimized to as close to 1 as possible.
        """
        # The gradient penalty is add to discriminator loss so that the it could be minimize. 
        difference = r_image - g_image
        difference *= tf.random.uniform((tf.shape(difference)[0], 1, 1, 1), 0.0, 1.0)
        between = g_image + difference
        
        with tf.GradientTape() as tape:
            tape.watch(between)
            gradient = tape.gradient(self.discriminator(between), between)
        gradient = tf.reduce_sum(tf.square(gradient), axis = (1, 2, 3))
        gradient = tf.sqrt(gradient + 1e-12)
        
        return tf.square(gradient - 1.0)
def define_generator(latent_size, img_size):
    input_ = tf.keras.Input((latent_size))
    layer = tf.keras.layers.Dense(256)(input_)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    
    layer = tf.keras.layers.Reshape((4, 4, 16))(layer)
    layer = tf.keras.layers.Conv2D(64, (3, 3), padding = "same")(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)

    level = int(math.log(img_size/4) / math.log(2))
    for _ in range(level):
        layer = tf.keras.layers.Conv2DTranspose(64, 5, strides = (2, 2), padding = "same")(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU()(layer)
        layer = tf.keras.layers.Conv2D(64, 5, padding = "same")(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.Conv2D(3, (7, 7), padding = "same")(layer)
    layer = tf.keras.layers.Activation("tanh")(layer)

    generator = tf.keras.Model(input_, layer, name = "generator")
    return generator

def define_discriminator(img_size):
    img = tf.keras.layers.Input((img_size, img_size, 3))
    layer = tf.keras.layers.Conv2D(64, (5, 5), strides = (2,2), padding = "same")(img)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    
    level = int(math.log(img_size/2 / 4) / math.log(2))

    for _ in range(level): 
        layer = tf.keras.layers.Conv2D(64, (5, 5), strides = (2,2), padding = "same")(layer)
        layer = tf.keras.layers.LayerNormalization()(layer)
        out = tf.keras.layers.LeakyReLU()(layer)

        layer = tf.keras.layers.Conv2D(64, (5, 5), padding = "same")(out)
        layer = tf.keras.layers.LayerNormalization()(layer + out)
        layer = tf.keras.layers.LeakyReLU()(layer)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(1)(layer)

    discriminator = tf.keras.Model(img, layer)

    return discriminator
