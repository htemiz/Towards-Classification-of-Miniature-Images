
from tensorflow.keras import metrics
from tensorflow.keras import losses, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,   concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.backend as K
from os.path import  dirname, abspath, basename
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import abstract_model


eps = 1.1e-6

normalize_batch = False
# pool_size = (6,6)
# pool_size = (2,2)
# pool_size = (4,4)
pool_size = (8,8)

class My_Model(abstract_model.model):

    def __init__(self, name='CNN', training_path=None, test_path=None):
        super().__init__(name, training_path, test_path)
        
    def __get_model__(self, mode='train' ):

        metrics = self.metrics if mode=='train' else self.test_metrics
        main_input = Input(shape=self.input_shape, name='main_input')
        x = self.data_augmentation(main_input)
        x = self.fn_normalization(x)
        feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(x)
        if self.max_pooling:
            feature_extraction = MaxPooling2D(pool_size, padding='valid')(feature_extraction)# feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)

        if self.normalize_batch:
            feature_extraction = BatchNormalization()(feature_extraction)

        feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)

        if self.max_pooling:
            feature_extraction = MaxPooling2D(pool_size, padding='valid')(feature_extraction)

            if self.normalize_batch:
                feature_extraction = BatchNormalization()(feature_extraction)

        feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)

        if self.max_pooling:
            feature_extraction = MaxPooling2D(pool_size, padding='valid')(feature_extraction)

            if self.normalize_batch:
                feature_extraction = BatchNormalization()(feature_extraction)

        x = Flatten()(feature_extraction)

        if self.normalize_batch:
            x = BatchNormalization()(x)
        # x = keras.layers.Dense(16, activation='relu')(x)
        output = Dense(4, activation='softmax')(x)

        model = Model(main_input, outputs=output, trainable=False)

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss, # CategoricalCrossentropy(from_logits=True), #
            metrics=metrics,
        )

        model.summary()

        return model
