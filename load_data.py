import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread


def load_data(nr_of_channels, batch_size=1, nr_target_train_imgs=None, nr_S1_train_imgs=None, nr_S2_train_imgs=None, nr_S3_train_imgs=None,
              nr_target_test_imgs=None, nr_S1_test_imgs=None, nr_S2_test_imgs=None, nr_S3_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0):

    trainT_path = os.path.join('data', subfolder, 'trainT')
    trainS1_path = os.path.join('data', subfolder, 'trainS1')
    trainS2_path = os.path.join('data', subfolder, 'trainS2')
    trainS3_path = os.path.join('data', subfolder, 'trainS3')
    testT_path = os.path.join('data', subfolder, 'testT')
    testS1_path = os.path.join('data', subfolder, 'testS1')
    testS2_path = os.path.join('data', subfolder, 'testS2')
    testS3_path = os.path.join('data', subfolder, 'testS3')
    trainT_image_names = os.listdir(trainT_path)
    if nr_target_train_imgs != None:
        trainT_image_names = trainT_image_names[:nr_target_train_imgs]

    trainS1_image_names = os.listdir(trainS1_path)
    if nr_S1_train_imgs != None:
        trainS1_image_names = trainS1_image_names[:nr_S1_train_imgs]
        
    trainS2_image_names = os.listdir(trainS2_path)
    if nr_S2_train_imgs != None:
        trainS2_image_names = trainS2_image_names[:nr_S2_train_imgs]
        
    trainS3_image_names = os.listdir(trainS3_path)
    if nr_S3_train_imgs != None:
        trainS3_image_names = trainS3_image_names[:nr_S3_train_imgs]
        
    testT_image_names = os.listdir(testT_path)
    if nr_target_test_imgs != None:
        testT_image_names = testT_image_names[:nr_target_test_imgs]

    testS1_image_names = os.listdir(testS1_path)
    if nr_S1_test_imgs != None:
        testS1_image_names = testS1_image_names[:nr_S1_test_imgs]

    testS2_image_names = os.listdir(testS2_path)
    if nr_S2_test_imgs != None:
        testS2_image_names = testS2_image_names[:nr_S2_test_imgs]
        
    testS3_image_names = os.listdir(testS3_path)
    if nr_S3_test_imgs != None:
        testS3_image_names = testS3_image_names[:nr_S3_test_imgs]
        
    if generator:
        return data_sequence(trainT_path, trainS1_path, trainS2_path, trainS3_path,  trainT_image_names, trainS1_image_names, trainS2_image_names, trainS3_image_names, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        trainT_images = create_image_array(trainT_image_names, trainT_path, nr_of_channels)
        trainS1_images = create_image_array(trainS1_image_names, trainS1_path, nr_of_channels)
        trainS2_images = create_image_array(trainS2_image_names, trainS2_path, nr_of_channels)
        trainS3_images = create_image_array(trainS3_image_names, trainS3_path, nr_of_channels)
        testT_images = create_image_array(testT_image_names, testT_path, nr_of_channels)
        testS1_images = create_image_array(testS1_image_names, testS1_path, nr_of_channels)
        testS2_images = create_image_array(testS2_image_names, testS2_path, nr_of_channels)
        testS3_images = create_image_array(testS3_image_names, testS3_path, nr_of_channels)
        return {"trainT_images": trainT_images, 
                "trainS1_images": trainS1_images, 
                "trainS2_images": trainS2_images,
                "trainS3_images": trainS3_images,
                "testT_images": testT_images, 
                "testS1_images": testS1_images, 
                "testS2_images": testS2_images,
                "testS3_images": testS3_images,
                "trainT_image_names": trainT_image_names,
                "trainS1_image_names": trainS1_image_names,
                "trainS2_image_names": trainS2_image_names,
                "trainS3_image_names": trainS3_image_names,
                "testT_image_names": testT_image_names,
                "testS1_image_names": testS1_image_names,
                "testS2_image_names": testS2_image_names,
                "testS3_image_names": testS3_image_names}
                


def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)
  
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
 #   array = array / 100
    return array


class data_sequence(Sequence):

    def __init__(self, trainT_path, trainS1_path, trainS2_path, trainS3_path, image_list_T, image_list_S1, image_list_S2, image_list_S3, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_T = []
        self.train_S1 = []
        self.train_S2 = []
        self.train_S3 = []
        for image_name in image_list_T:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_T.append(os.path.join(trainT_path, image_name))
        for image_name in image_list_S1:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_S1.append(os.path.join(trainS1_path, image_name))
        for image_name in image_list_S2:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_S2.append(os.path.join(trainS2_path, image_name))
        for image_name in image_list_S3:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_S3.append(os.path.join(trainS3_path, image_name))
    def __len__(self):
        return int(max(len(self.train_T), len(self.train_S1), len(self.train_S2), len(self.train_S3)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
#        if idx >= min(len(self.train_A), len(self.train_B)):
#            # If all images soon are used for one domain,
#            # randomly pick from this domain
#            if len(self.train_A) <= len(self.train_B):
#                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
#                batch_A = []
#                for i in indexes_A:
#                    batch_A.append(self.train_A[i])
#                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
#            else:
#                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
#                batch_B = []
#                for i in indexes_B:
#                    batch_B.append(self.train_B[i])
#                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
#        else:
        batch_T = self.train_T[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_S1 = self.train_S1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_S2 = self.train_S2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_S3 = self.train_S3[idx * self.batch_size:(idx + 1) * self.batch_size]
        real_images_T = create_image_array(batch_T, '', 3)
        real_images_S1 = create_image_array(batch_S1, '', 3)
        real_images_S2 = create_image_array(batch_S2, '', 3)
        real_images_S3 = create_image_array(batch_S3, '', 3)
        return real_images_T, real_images_S1, real_images_S2, real_images_S3  # input_data, target_data


if __name__ == '__main__':
    load_data()
