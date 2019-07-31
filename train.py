import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import keras
from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D, concatenate
from keras.utils.conv_utils import normalize_data_format
from keras_contrib.layers.normalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Container
from keras.losses import hinge
from collections import OrderedDict
from scipy.misc import imsave, toimage  # has depricated
import numpy as np
import random
import datetime
import time
import json
import math
import csv
import sys
import scipy
#import os
import keras.backend as K
import tensorflow as tf
import load_data
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
np.random.seed(seed=12345)


class DiamondGAN():
    def __init__(self, lr_D=2e-4, lr_G=2e-4, img_shape=(240, 240, 3), domain_shape = (240, 240, 1),
                 date_time_string_addition='', image_folder='MR'):
        self.img_shape = img_shape
        self.hinge = hinge
 #       self.gram_loss = gram_loss
        self.domain_shape = domain_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 8.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 8.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 5
        self.epochs = 100  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 25

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True
        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False
        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False
        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        # Tweaks
        self.REAL_LABEL = 0.95  # Use e.g. 0.9 to avoid training the discriminators to zero loss
        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition
        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)
        ## S = source domain, T = target domain
        # ======= Discriminator model ========== **** domain matrix as a part of the input???? ****
        if self.use_multiscale_discriminator:
            D_S = self.modelMultiScaleDiscriminator()  #source domain
            D_T = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
        else:
            D_S = self.modelDiscriminator()
            D_T = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images

        # Discriminator builds
        image_S = Input(shape=self.img_shape)
        image_T = Input(shape=self.img_shape)
        guess_S = D_S(image_S)
        guess_T = D_T(image_T)
        self.D_S = Model(inputs=image_S, outputs=guess_S, name='D_S_model')
        self.D_T = Model(inputs=image_T, outputs=guess_T, name='D_T_model')

        # self.D.summary()
        self.D_S.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_T.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        
        # Use containers to avoid falsy keras error about weight descripancies
        self.D_S_static = Container(inputs=image_S, outputs=guess_S, name='D_S_static_model')
        self.D_T_static = Container(inputs=image_T, outputs=guess_T, name='D_T_static_model')

#######################################################################################################
        # ======= Generator model ==========
        # Do not update discriminator weights during generator training
        self.D_S_static.trainable = False
        self.D_T_static.trainable = False
        # Generators
        self.G_S2T = self.modelGenerator(name='G_S2T_model')
        self.G_T2S = self.modelGenerator(name='G_T2S_model')
        # self.G.summary()
        # Generator builds
        real_S = Input(shape=self.img_shape, name='real_S')
        real_T = Input(shape=self.img_shape, name='real_T')
        domain_matrix = Input(shape=self.domain_shape, name='domain_matrix')  # the source domain matrix to identify the input modality. It is produced by spacially replicating the domain-mask vector
        
        synthetic_T = self.G_S2T([real_S, domain_matrix])  # source to target
        synthetic_S = self.G_T2S([real_T, domain_matrix])
        
        dS_guess_synthetic = self.D_S_static(synthetic_S)
        dT_guess_synthetic = self.D_T_static(synthetic_T)
        reconstructed_S = self.G_T2S([synthetic_T, domain_matrix])
        reconstructed_T = self.G_S2T([synthetic_S, domain_matrix])
        
        model_outputs = [reconstructed_S, reconstructed_T]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]


        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(self.lse)
                compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dS_guess_synthetic[i])
                model_outputs.append(dT_guess_synthetic[i])
        else:
            model_outputs.append(dS_guess_synthetic)
            model_outputs.append(dT_guess_synthetic)

        self.G_model = Model(inputs=[real_S, real_T, domain_matrix], ### 
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

        # ======= Data ==========
        # Use 'None' to fetch all available images
        nr_target_train_imgs = None # 
        nr_S1_train_imgs = None  # source domain images
        nr_S2_train_imgs = None
        nr_S3_train_imgs = None
        nr_target_test_imgs = None
        nr_S1_test_imgs = None
        nr_S2_test_imgs = None
        nr_S3_test_imgs = None
        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if self.use_data_generator:
            self.data_generator = load_data.load_data(
                nr_of_channels=self.batch_size, generator=True, subfolder=image_folder)

            # Only store test images
            nr_target_train_imgs = 0
            nr_S1_train_imgs = 0
            nr_S2_train_imgs = 0
            nr_S3_train_imgs = 0
        data = load_data.load_data(nr_of_channels=self.channels,
                                   batch_size=self.batch_size,
                                   nr_target_train_imgs=nr_target_train_imgs,
                                   nr_S1_train_imgs=nr_S1_train_imgs,
                                   nr_S2_train_imgs=nr_S2_train_imgs,
                                   nr_S3_train_imgs=nr_S3_train_imgs,
                                   nr_target_test_imgs=nr_target_test_imgs,
                                   nr_S1_test_imgs=nr_S1_test_imgs,
                                   nr_S2_test_imgs=nr_S2_test_imgs,
                                   nr_S3_test_imgs=nr_S3_test_imgs,
                                   subfolder=image_folder)

        self.T_train = data["trainT_images"]
        self.S1_train = data["trainS1_images"]
        self.S2_train = data["trainS2_images"]
        self.S3_train = data["trainS3_images"]
        self.T_test = data["testT_images"]
        self.S1_test = data["testS1_images"]
        self.S2_test = data["testS2_images"]
        self.S3_test = data["testS3_images"]
        self.testT_image_names = data["testT_image_names"]
        self.testS1_image_names = data["testS1_image_names"]
        self.testS2_image_names = data["testS2_image_names"]
        self.testS3_image_names = data["testS3_image_names"]
        if not self.use_data_generator:
            print('Data has been loaded')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writeMetaDataToJSON()

        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))

        # ======= Initialize training ==========
        sys.stdout.flush()
        #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)
        self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
#        self.load_model_and_generate_synthetic_images()

#===============================================================================
# Architecture functions / blocks

    def ck(self, x, k, use_normalization):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        input_domain = Input(shape=self.domain_shape)
      
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        d = ReflectionPadding2D((3, 3))(input_domain)
       
        x = self.c7Ak(x, 48)
        d = self.c7Ak(d, 48)
        # Layer 2
        x = self.dk(x, 72)
        d = self.dk(d, 72)
        # Layer 3
        x = self.dk(x, 144)
        d = self.dk(d, 144)
        
        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)
            d = self.dk(d, 256)
     #   x = concatenate([x, d], axis = -1)
        # Layer 4-12: Residual layer
        
        for _ in range(4, 13):
            x = self.Rk(x)
            d = self.Rk(d)
        x = concatenate([x, d], axis = -1)
        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)

        # Layer 13
        x = self.uk(x, 72)
        # Layer 14
        x = self.uk(x, 48)
        d = self.uk(d, 48)
        x = ReflectionPadding2D((3, 3))(x)
   #     d = ReflectionPadding2D((3, 3))(d)
        x = Conv2D(3, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=[input_img, input_domain], outputs=x, name=name)

#==============================================================================
#===============================================================================
# Training
    def train(self, epochs, batch_size=1, save_interval=1):
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
                # Generate batch of synthetic images
            synthetic_images_T = self.G_S2T.predict([real_images_S, domain_matrix])
            synthetic_images_S = self.G_T2S.predict([real_images_T, domain_matrix])
            synthetic_images_S = synthetic_pool_S.query(synthetic_images_S)
            synthetic_images_T = synthetic_pool_T.query(synthetic_images_T)
            
            for _ in range(self.discriminator_iterations):
                DS_loss_real = self.D_S.train_on_batch(x=real_images_S, y=ones)
                DT_loss_real = self.D_T.train_on_batch(x=real_images_T, y=ones)
                DS_loss_synthetic = self.D_S.train_on_batch(x=synthetic_images_S, y=zeros)
                DT_loss_synthetic = self.D_T.train_on_batch(x=synthetic_images_T, y=zeros)

                if self.use_multiscale_discriminator:
                    DS_loss = sum(DS_loss_real) + sum(DS_loss_synthetic)
                    DT_loss = sum(DT_loss_real) + sum(DT_loss_synthetic)
                    print('DS_losses: ', np.add(DS_loss_real, DS_loss_synthetic))
                    print('DT_losses: ', np.add(DT_loss_real, DT_loss_synthetic))
                else:
                    DS_loss = DS_loss_real + DS_loss_synthetic
                    DT_loss = DT_loss_real + DT_loss_synthetic
                D_loss = DS_loss + DT_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_S, real_images_T]  # Compare reconstructed images to real images
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones)
                target_data.append(ones)

          

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_S, real_images_T, domain_matrix], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gS_d_loss_synthetic = G_loss[1]
            gT_d_loss_synthetic = G_loss[2]
            reconstruction_loss_S = G_loss[3]
            reconstruction_loss_T = G_loss[4]

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_S, decay_D)
                self.update_lr(self.D_T, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store some training data
            DS_losses.append(DS_loss)
            DT_losses.append(DT_loss)
            gS_d_losses_synthetic.append(gS_d_loss_synthetic)
            gT_d_losses_synthetic.append(gT_d_loss_synthetic)
            gS_losses_reconstructed.append(reconstruction_loss_S)
            gT_losses_reconstructed.append(reconstruction_loss_T)

            GS_loss = gS_d_loss_synthetic + reconstruction_loss_S
            GT_loss = gT_d_loss_synthetic + reconstruction_loss_T
            D_losses.append(D_loss)
            GS_losses.append(GS_loss)
            GT_losses.append(GT_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_S + reconstruction_loss_T
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('DS_loss:', DS_loss)
            print('DT_loss:', DT_loss)

            if loop_index % 20 == 0:
                # Save temporary images continously
             #   self.save_tmp_images(real_images_S, real_images_T, synthetic_images_S, synthetic_images_T)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)


        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DS_losses = []
        DT_losses = []
        gS_d_losses_synthetic = []
        gT_d_losses_synthetic = []
        gS_losses_reconstructed = []
        gT_losses_reconstructed = []

        GS_losses = []
        GT_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_S = ImagePool(self.synthetic_pool_size)
        synthetic_pool_T = ImagePool(self.synthetic_pool_size)
        # self.saveImages('(init)')
        # labels
        if self.use_multiscale_discriminator:
            label_shape1 = (batch_size,) + self.D_S.output_shape[0][1:]
            label_shape2 = (batch_size,) + self.D_S.output_shape[1][1:]
            #label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            #zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (batch_size,) + self.D_S.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_images_T = images[0]
                    real_images_S1 = images[1]
                    real_images_S2 = images[2]
                    real_images_S3 = images[3]
                    if len(real_images_T.shape) == 3:
                        real_images_T = real_images_T[:, :, :, np.newaxis]
                        real_images_S1 = real_images_S1[:, :, :, np.newaxis]
                        real_images_S2 = real_images_S2[:, :, :, np.newaxis]
                        real_images_S3 = real_images_S3[:, :, :, np.newaxis]
                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator.__len__())

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D, loop_index)
                        self.saveModel(self.G, loop_index)
                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index += 1

            else:  # Train with all data in cache
                T_train = self.T_train
                S1_train = self.S1_train
                S2_train = self.S2_train
                S3_train = self.S3_train
                random_order_T = np.random.randint(len(T_train), size=len(T_train))
                
                epoch_iterations = len(random_order_T)
 #               min_nr_imgs = len(random_order_A)
############################   KEY COMPONENT  ###################
                domain_matrix_all = np.load('domain_matrix.npy')
                domain_dictionary = np.load('domain_dictionary.npy')
                for loop_index in range(0, epoch_iterations, batch_size):
                    indexes_T = random_order_T[loop_index:loop_index + batch_size]
                    indexes_S1 = indexes_T
                    indexes_S2 = indexes_T
                    indexes_S3 = indexes_T
                    sys.stdout.flush()
                    real_images_T = T_train[indexes_T]
                    
                   # 
                    real_images_S1 = S1_train[indexes_S1]
                    real_images_S2 = S2_train[indexes_S2]
                    real_images_S3 = S3_train[indexes_S3]
                    if len(real_images_S1.shape) == 3:
                        real_images_T = real_images_T[:, :, :, np.newaxis]
                        real_images_S1 = real_images_S1[:, :, :, np.newaxis]
                        real_images_S2 = real_images_S2[:, :, :, np.newaxis]
                        real_images_S3 = real_images_S3[:, :, :, np.newaxis]
                    # Run all training steps
############ generating multi-channel input data based on the varied domain input
                   # domain_dictionary = [[1, 0, 0],  [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
                    #domain_matrix = np.zeros(np.shape(target_images), dtype = 'float64')
                    real_images_T = np.concatenate((real_images_T, real_images_T, real_images_T), axis = -1)
                    print(np.shape(real_images_T))
                    for dd in range(np.shape(domain_dictionary)[0]):
                        real_images_S = np.concatenate((real_images_S1, real_images_S2, real_images_S3), axis = -1)
                        
                        if dd == 0: 
                            real_images_S[..., 1] = -1
                            real_images_S[..., 2] = -1
                        if dd == 1: 
                            real_images_S[..., 0] = -1
                            real_images_S[..., 2] = -1
                        if dd == 2: 
                            real_images_S[..., 0] = -1
                            real_images_S[..., 1] = -1
                        if dd == 3:
                            real_images_S[..., 2] = -1
                        if dd == 4:
                            real_images_S[..., 0] = -1
                        if dd == 5:
                            real_images_S[..., 1] = -1
                    
                        domain_matrix = np.tile(domain_matrix_all[:, :, dd:dd+1], (batch_size, 1, 1, 1))
                    # Run all training steps
                        run_training_iteration(loop_index, epoch_iterations)

            #================== within epoch loop end ==========================
            if epoch % 1 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_S, epoch)
                self.saveModel(self.D_T, epoch)
                self.saveModel(self.G_S2T, epoch)
                self.saveModel(self.G_T2S, epoch)

            training_history = {
                'DS_losses': DS_losses,
                'DT_losses': DT_losses,
                'gS_d_losses_synthetic': gS_d_losses_synthetic,
                'gT_d_losses_synthetic': gT_d_losses_synthetic,
                'gS_losses_reconstructed': gS_losses_reconstructed,
                'gT_losses_reconstructed': gT_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()

#===============================================================================
# Help functions
    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):  # normal L1 loss 
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss
#    def combined_loss(self, y_true, y_pred):
#        l1 = self.cycle_loss(y_true, y_pred)
#        def gram_matrix(x):
#            #assert K.ndim(x) == 3
#            if K.image_data_format() == 'channels_first':
#                features = K.batch_flatten(x)
#            else:
#                features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1, 3)))
#            gram = K.dot(features, K.transpose(features))
#            return gram 
#        
##        assert K.ndim(y_true) == 3
##        assert K.ndim(y_pred) == 3
#        S = gram_matrix(y_true)
#        C = gram_matrix(y_pred)
#        channels = 1
#        size = 240 * 240
#        return 0.5*K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2)) +  l1
#    
    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        synthetic = synthetic.clip(min=0)
        reconstructed = reconstructed.clip(min=0)

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        toimage(image, cmin=0, cmax=1).save(path_name)

#    def saveImages(self, epoch, real_image_S, real_image_T, num_saved_images=1):
#        directory = os.path.join('images', self.date_time)
#        if not os.path.exists(os.path.join(directory, 'S')):
#            os.makedirs(os.path.join(directory, 'S'))
#            os.makedirs(os.path.join(directory, 'T'))
#            os.makedirs(os.path.join(directory, 'Stest'))
#            os.makedirs(os.path.join(directory, 'Ttest'))
#
#        testString = ''
#
#        real_image_Ab = None
#        real_image_Ba = None
#        for i in range(num_saved_images + 1):
#            if i == num_saved_images:
#                real_image_A = self.A_test[0]
#                real_image_B = self.B_test[0]
#                real_image_A = np.expand_dims(real_image_A, axis=0)
#                real_image_B = np.expand_dims(real_image_B, axis=0)
#                testString = 'test'
#                if self.channels == 1:  # Use the paired data for MR images
#                    real_image_Ab = self.B_test[0]
#                    real_image_Ba = self.A_test[0]
#                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
#                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)
#            else:
#                #real_image_A = self.A_train[rand_A_idx[i]]
#                #real_image_B = self.B_train[rand_B_idx[i]]
#                if len(real_image_A.shape) < 4:
#                    real_image_A = np.expand_dims(real_image_A, axis=0)
#                    real_image_B = np.expand_dims(real_image_B, axis=0)
#                if self.channels == 1:  # Use the paired data for MR images
#                    real_image_Ab = real_image_B  # self.B_train[rand_A_idx[i]]
#                    real_image_Ba = real_image_A  # self.A_train[rand_B_idx[i]]
#                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
#                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)
#
#            synthetic_image_B = self.G_A2B.predict(real_image_A)
#            synthetic_image_A = self.G_B2A.predict(real_image_B)
#            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
#            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)
#
#            self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
#                                 'images/{}/{}/epoch{}_sample{}.png'.format(
#                                     self.date_time, 'A' + testString, epoch, i))
#            self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
#                                 'images/{}/{}/epoch{}_sample{}.png'.format(
#                                     self.date_time, 'B' + testString, epoch, i))

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.S1_train), len(self.T_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
       
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)


#===============================================================================
# Save and load

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_models/{}/{}_weights_epoch_{}.hdf5'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of T train examples': len(self.T_train),
            'number of S1 train examples': len(self.S1_train),
            'number of T test examples': len(self.T_test),
            'number of S1 test examples': len(self.S1_test),
        })

        with open('images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
        path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
        path_to_weights = os.path.join('generate_images', 'models', '{}.hdf5'.format(model.name))
        #model = model_from_json(path_to_model)
        model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self):
        response = input('Are you sure you want to generate synthetic images instead of training? (y/n): ')[0].lower()
        if response == 'y':
            domain_matrix_all = np.load('domain_matrix.npy')
            domain_dictionary = np.load('domain_dictionary.npy')
            self.load_model_and_weights(self.G)
            # save function
            def save_image(image, name, domain):
                image = image[:, :, 0]
                scipy.misc.imsave(os.path.join('generate_images', 'synthetic_images', domain, name), image)
                
############ generating multi-channel input data based on the varied domain input
                   # domain_dictionary = [[1, 0], [0, 1]]
            S1_test = self.S1_test
            if len(S1_test.shape) == 3:
                S1_test = self.S1_test[:, :, :, np.newaxis]
                S2_test = self.S2_test[:, :, :, np.newaxis]
                S3_test = self.S3_test[:, :, :, np.newaxis]
            #input_all = np.concatenate((B_test, C_test), axis = -1)
            for dd in range(np.shape(domain_dictionary)[0]):
        
                domain_matrix = np.tile(domain_matrix_all[:, :, dd:dd+1], (np.shape(S1_test)[0], 1, 1, 1))
                if dd == 0:
                    input_images = S1_test
                    synthetic_images = self.G_S2T.predict([input_images, domain_matrix])
                    for i in range(len(synthetic_images)):
                # Get the name from the image it was conditioned on
                        name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
                        synt_A = synthetic_images[i]
                        save_image(synt_A, name, 'T_s1')
                if dd == 1:
                    input_images = S2_test
                    synthetic_images = self.G_S2T.predict([input_images, domain_matrix])
                    for i in range(len(synthetic_images)):
                # Get the name from the image it was conditioned on
                        name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
                        synt_A = synthetic_images[i]
                        save_image(synt_A, name, 'T_s2') 
                if dd == 2:
                    input_images = S3_test
                    synthetic_images = self.G_S2T.predict([input_images, domain_matrix])
                    for i in range(len(synthetic_images)):
                # Get the name from the image it was conditioned on
                        name = self.testA_image_names[i].strip('.png') + '_synthetic.png'
                        synt_A = synthetic_images[i]
                        save_image(synt_A, name, 'T_s3') 
                    

            
# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


if __name__ == '__main__':
    GAN = DiamondGAN()
