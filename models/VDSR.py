"""
VDSR-19 model tailored for classifying Ottoman Miniature Images. This is a DeepSR file.

To run this model, e.g., training:

--python.exe -m DeepSR.DeepSR --modelfile VDSR.py --train

Please refer to the documentation of DeepSR from the following address for other command instructions and setting:
    https://github.com/htemiz/DeepSR
"""

"""
VDSR model
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf

"""
import numpy as np
from keras import metrics
from keras import losses
from keras.layers import Input, Dense, Activation, add
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from os.path import dirname, abspath, basename
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import LearningRateScheduler

eps = 1.1e-6

settings = \
    {
        'augment': [],  #  any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ]
        'backend': 'tensorflow',
        'batchsize': 256,
        'channels': 3,
        'colormode': 'RGB',  # 'YCbCr' or 'RGB'
        'crop': 0,
        'crop_test': 6,
        'decay': 0.9,
        'decimation': 'bicubic',
        'dilation_rate':(1,1),
        'epatience': 25,
        'epoch': 25,
        'inputsize': 41,  #
        'interp_compare': '',
        'interp_up': 'bicubic',
        'kernel_initializer': 'he_normal',
        'lrate': 0.0001,
        'lrpatience': 15,
        'lrfactor': 0.5,
        'metrics': 'PSNR',
        'minimumlrate': 1e-7,
        'modelname': basename(__file__).split('.')[0],
        'noise': '',
        'normalization': ['divide', 255.0],  # ['standard', "53.28741141", "40.73203139"],
        'normalizeback': False,
        'normalizeground': False,
        'outputdir': '',
        'scale': 2,
        'seed': 19,
        'shuffle': False,
        'stride': 41,  # the same with input size, to have no overlap as is in the paper.
        'target_channels': 3,
        'target_cmode': 'RGB',
        'testpath': [r'D:\calisma\projeler\ottoman\dataset\test'],
        'traindir': r"D:\calisma\projeler\ottoman\dataset\train",
        'upscaleimage': True,
        'valdir': r'D:\calisma\projeler\ottoman\dataset\val',
        'weightpath': '',
        'workingdir': '',
    }


def build_model(self, testmode=False):
    # learning rate schedule

    from tensorflow.math import log
    from tensorflow import constant

    def tf_log10(x):
        numerator = log(x)
        denominator = log(constant(10, dtype=numerator.dtype))

        return numerator / denominator

    def PSNR(y_true, y_pred):
        from keras.backend import mean as Mean, square as Square

        max_pixel = 1.0
        return 10.0 * tf_log10((max_pixel ** 2) / (Mean(Square(y_pred - y_true))))



    if testmode:
        input_size = None
    else:
        input_size = self.inputsize

    input_shape = (input_size, input_size, self.channels)

    input_img = Input(shape=input_shape, name='main_input')

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(input_img)
    model = self.apply_activation(model, self.activation, self.activation + "_001")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_002")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_003")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_004")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_005")

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_006")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_007")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_008")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_009")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_010")

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_011")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_012")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_013")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_014")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_015")

    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_016")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_017")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.activation, self.activation + "_018")
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)(model)
    model = self.apply_activation(model, self.lactivation, self.lactivation + "_019")
    model = Conv2D(self.channels, (3, 3), padding='same',  kernel_initializer=self.kernel_initializer)(model)
    res_img = model

    output_img = add([res_img, input_img])

    model = Model(input_img, output_img)


    # Compile model
    #sgd = SGD(lr=self.lrate, momentum=0.9)  # used in paper
    adam = Adam(self.lrate, self.decay)  # my approach
    
    model.compile( loss=losses.mean_squared_error, optimizer=adam) #, metrics=[PSNR, 'accuracy'])
    # learning schedule callback

    # model.summary()

    return model

"""

def callbacks(self):
    callbacks_list=[]

    def step_decay(epoch):
        from tensorflow.math import pow, floor
        initial_lrate = self.lrate  # 0.1, in paper
        drop = 0.5  # 0.5, in paper
        epochs_drop = 20.0  # 20, in paper
        lrate = initial_lrate * pow(drop, floor((1 + epoch) / epochs_drop))
        return np.float(lrate)

    lrate = LearningRateScheduler(step_decay)

    callbacks_list.append(lrate)

    return callbacks_list
"""