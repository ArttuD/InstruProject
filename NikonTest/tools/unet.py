from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D,Conv2DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50V2
import tensorflow as tf
import tensorflow.keras.layers as tkl
from tools.losses import binary_dice,focal_loss


def Unet(num_class, image_size):
    """Semantic segmentation using Unet"""
    # encoder
    inputs = Input(shape=[image_size, image_size, 3])
    conv1 = Conv2D(12, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(12, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # decoder
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(12, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(12, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(12, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(12, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv2D(num_class, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = focal_loss(alpha=0.25, gamma=2), metrics = ['accuracy'])

    return model

def batch_norm(x,trainable=False):
    x = tkl.BatchNormalization(trainable=trainable)(x)
    x = tkl.ReLU()(x)
    return x

def conv_block(x,filters,kernel,pad=False,trainable=False):

    # conv block
    x = Conv2D(filters,kernel,padding='valid',trainable=trainable,bias_initializer=tf.constant_initializer(value=0.01))(x)
    x = tkl.BatchNormalization(trainable=trainable)(x)
    x = tkl.ReLU()(x)
    if pad:
        x = tkl.ZeroPadding2D((1,1))(x)

    return x

def res_block(x,filters,first=False,trainable=False):

    
    block_pre = batch_norm(x)
    block = conv_block(block_pre,filters[0],(2,2),True)
    block = conv_block(block,filters[0],(2,2))
    block = Conv2D(filters[1],(1,1),trainable=trainable,bias_initializer=tf.constant_initializer(value=0.01))(block)
    # skip
    if first:
        block_skip = Conv2D(filters[1],(1,1),trainable=trainable,bias_initializer=tf.constant_initializer(value=0.01))(block_pre)
    else:
        block_skip = x
    # Add
    block_add = tkl.Add()([block_skip,block])

    return block_add

def Unet_ResNet(num_class, image_size,trainable=False,finetune=False,loss='focal'):
    """Unet with ResNet50 encoder and decoder. Uses pretrained network for encoder
       and freezes it aka transfer learning"""
    # load pretrained ResNet 50
    encoder = ResNet50V2(
                input_shape=(image_size,image_size,3),
                include_top=False,
                weights='imagenet')
    
    encoder.trainabe = False

    names = [i.name for i in encoder.layers]

    # decoder
    # conv4
    conv5_up = UpSampling2D(size=(1,1))(encoder.layers[names.index('post_relu')].output)
    conv4 = encoder.layers[names.index('conv4_block6_2_relu')].output
    conv4_up_encoder = concatenate([conv5_up,conv4],axis=3)

    conv4_block_1 = res_block(conv4_up_encoder,[256,1024],first=True,trainable=trainable)
    conv4_block_2 = res_block(conv4_block_1,[256,1024],trainable=trainable)
    conv4_block_3 = res_block(conv4_block_2,[256,1024],trainable=trainable)
    conv4_block_4 = res_block(conv4_block_3,[256,1024],trainable=trainable)
    conv4_block_5 = res_block(conv4_block_4,[256,1024],trainable=trainable)
    conv4_block_6 = res_block(conv4_block_5,[256,1024],trainable=trainable)

    # conv3
    conv4_up = UpSampling2D((2,2))(conv4_block_4)
    conv3 = encoder.layers[names.index('conv3_block4_2_relu')].output
    conv3_up_encoder = concatenate([conv4_up,conv3],axis=3)

    conv3_block_1 = res_block(conv3_up_encoder,[128,512],first=True,trainable=trainable)
    conv3_block_2 = res_block(conv3_block_1,[128,512],trainable=trainable)
    conv3_block_3 = res_block(conv3_block_2,[128,512],trainable=trainable)
    conv3_block_4 = res_block(conv3_block_3,[128,512],trainable=trainable)

    # conv2
    conv3_up = UpSampling2D((2,2))(conv3_block_4)
    conv2 = encoder.layers[names.index('conv2_block3_2_relu')].output
    conv2_up_encoder = concatenate([conv3_up,conv2],axis=3) 

    conv2_block_1 = res_block(conv2_up_encoder,[64,256],first=True,trainable=trainable)
    conv2_block_2 = res_block(conv2_block_1,[64,256],trainable=trainable)
    conv2_block_3 = res_block(conv2_block_2,[64,256],trainable=trainable)


    # prediction
    conv1_1_up = UpSampling2D((2,2))(conv2_block_3)
    conv1_1_block_1 = res_block(conv1_1_up,[32,128],first=True,trainable=trainable)
    conv1_1_block_2 = res_block(conv1_1_block_1,[32,128],trainable=trainable)
    conv1_1_block_3 = res_block(conv1_1_block_2,[32,128],trainable=trainable)

    conv1_2_up = UpSampling2D((2,2))(conv1_1_block_3)
    conv1_2_block_1 = res_block(conv1_2_up,[16,64],first=True,trainable=trainable)
    conv1_2_block_2 = res_block(conv1_2_block_1,[16,64],trainable=trainable)
    conv1_2_block_3 = res_block(conv1_2_block_2,[16,64],trainable=trainable)

    conv1_3_up = UpSampling2D((2,2))(conv1_2_block_3)
    conv1_3_block_1 = res_block(conv1_3_up,[8,32],first=True,trainable=trainable)
    conv1_3_block_2 = res_block(conv1_3_block_1,[8,32],trainable=trainable)
    conv1_3_block_3 = res_block(conv1_3_block_2,[8,32],trainable=finetune)

    conv_result = Conv2D(num_class, 1, activation = 'sigmoid')(conv1_3_block_3)
    
    model = Model(inputs = encoder.input, outputs = conv_result)
    if loss =='binary':
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    elif loss == 'dice':
        model.compile(optimizer = Adam(lr = 1e-4), loss = binary_dice, metrics = ['accuracy'])
    elif loss == 'focal':
        model.compile(optimizer = Adam(lr = 1e-4), loss = focal_loss(alpha=0.25, gamma=2), metrics = ['accuracy'])

    return model

def standard_block(img_in,filters,name,kernel=(3,3)):
    x = Conv2D(filters,kernel,activation='relu',name='block_{}_stage_1'.format(name),padding='same')(img_in)
    x = Dropout(0.5,name='dropout_{}_name_1'.format(name))(x)
    x = Conv2D(filters,kernel,activation='relu',name='block_{}_stage_2'.format(name),padding='same')(img_in)
    x = Dropout(0.5,name='dropout_{}_name_1'.format(name))(x)
    return x

def Unet_plusplus(num_class,img_shape,deep_supervision=False,loss='dice'):
    """
    Unet++. Implementation based on https://github.com/MrGiovanni/UNetPlusPlus
    """
    filters = [32,64,128,256,512]
    net_in = Input(shape=(img_shape,img_shape,3),name='input')

    # block 1,1 
    conv1_1 = standard_block(net_in, name='11', filters=filters[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    # block 2,1
    conv2_1 = standard_block(pool1, name='21', filters=filters[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    # block 1,2
    up1_2 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = standard_block(conv1_2, name='12', filters=filters[0])

    # block 3,1
    conv3_1 = standard_block(pool2, name='31', filters=filters[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    # block 2,2
    up2_2 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = standard_block(conv2_2, name='22', filters=filters[1])

    # block 1,3
    up1_3 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = standard_block(conv1_3, name='13', filters=filters[0])

    # block 4,1
    conv4_1 = standard_block(pool3, name='41', filters=filters[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    # block 3,2
    up3_2 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = standard_block(conv3_2, name='32', filters=filters[2])

    # block 2,3
    up2_3 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = standard_block(conv2_3, name='23', filters=filters[1])

    # block 1,4
    up1_4 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = standard_block(conv1_4, name='14', filters=filters[0])

    # block 5,1
    conv5_1 = standard_block(pool4, name='51', filters=filters[4])
 
    # block 4,2
    up4_2 = Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_block(conv4_2, name='42', filters=filters[3])

    # block 3,3
    up3_3 = Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = standard_block(conv3_3, name='33', filters=filters[2])

    # block 2,4
    up2_4 = Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = standard_block(conv2_4, name='24', filters=filters[1])

    # block 1,5
    up1_5 = Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = standard_block(conv1_5, name='15', filters=filters[0])

    # for deep supervision
    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(inputs=net_in, outputs=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(inputs=net_in, outputs=[nestnet_output_4])

    if loss =='binary':
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    elif loss == 'dice':
        model.compile(optimizer = Adam(lr = 1e-4), loss = binary_dice, metrics = ['accuracy'])
    elif loss == 'focal':
        model.compile(optimizer = Adam(lr = 1e-4), loss = focal_loss(alpha=0.25, gamma=2), metrics = ['accuracy'])

    return model
