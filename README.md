# EIP_Session3
Assignment 3
# Base network accuracy

After running the base network for 50 epochs the validation accuracy is - 82.66

![image](https://github.com/sridevibonthu/EIP_Session3/blob/master/basenwaccuracy.JPG)

# My network

Here I have used Separable Convolution, Batch Normalization, Dropout and removed Dense layers at the end which were present in the vgg6 model architecture. this has beaten base network by crossing 83% of validation accuracy. But the parameters reached 100,021. I have removed biases and made it 99,243 in my second network. I have copied both the networks here.

```python
model = Sequential()
model.add(SeparableConvolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3))) #(outputsize - 32, Receptive Field - 3)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(48, 3, 3)) #(30, 5)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(MaxPooling2D(pool_size=(2, 2))) #(15, 6)
model.add(Dropout(0.2))

model.add(SeparableConvolution2D(96, 3, 3, border_mode='same')) #(15, 10)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(96, 3, 3)) #(13, 14)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(MaxPooling2D(pool_size=(2, 2))) #(6, 16)
model.add(Dropout(0.2))

model.add(SeparableConvolution2D(192, 3, 3, border_mode='same')) #(6, 24)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(192, 3, 3)) #(4, 32)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(MaxPooling2D(pool_size=(2, 2))) #(2, 36)
model.add(Dropout(0.2))

model.add(SeparableConvolution2D(96, 2, 2)) #(1, 44)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(num_classes, 1, 1)) #(1, 44)
model.add(Activation('relu'))

model.add(Flatten())
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()```

```python
Model: "sequential_12"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_73 (Separab (None, 32, 32, 48)        219       
_________________________________________________________________
activation_98 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_64 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
dropout_77 (Dropout)         (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_74 (Separab (None, 30, 30, 48)        2784      
_________________________________________________________________
activation_99 (Activation)   (None, 30, 30, 48)        0         
_________________________________________________________________
batch_normalization_65 (Batc (None, 30, 30, 48)        192       
_________________________________________________________________
dropout_78 (Dropout)         (None, 30, 30, 48)        0         
_________________________________________________________________
max_pooling2d_34 (MaxPooling (None, 15, 15, 48)        0         
_________________________________________________________________
dropout_79 (Dropout)         (None, 15, 15, 48)        0         
_________________________________________________________________
separable_conv2d_75 (Separab (None, 15, 15, 96)        5136      
_________________________________________________________________
activation_100 (Activation)  (None, 15, 15, 96)        0         
_________________________________________________________________
batch_normalization_66 (Batc (None, 15, 15, 96)        384       
_________________________________________________________________
dropout_80 (Dropout)         (None, 15, 15, 96)        0         
_________________________________________________________________
separable_conv2d_76 (Separab (None, 13, 13, 96)        10176     
_________________________________________________________________
activation_101 (Activation)  (None, 13, 13, 96)        0         
_________________________________________________________________
batch_normalization_67 (Batc (None, 13, 13, 96)        384       
_________________________________________________________________
dropout_81 (Dropout)         (None, 13, 13, 96)        0         
_________________________________________________________________
max_pooling2d_35 (MaxPooling (None, 6, 6, 96)          0         
_________________________________________________________________
dropout_82 (Dropout)         (None, 6, 6, 96)          0         
_________________________________________________________________
separable_conv2d_77 (Separab (None, 6, 6, 192)         19488     
_________________________________________________________________
activation_102 (Activation)  (None, 6, 6, 192)         0         
_________________________________________________________________
batch_normalization_68 (Batc (None, 6, 6, 192)         768       
_________________________________________________________________
dropout_83 (Dropout)         (None, 6, 6, 192)         0         
_________________________________________________________________
separable_conv2d_78 (Separab (None, 4, 4, 192)         38784     
_________________________________________________________________
activation_103 (Activation)  (None, 4, 4, 192)         0         
_________________________________________________________________
batch_normalization_69 (Batc (None, 4, 4, 192)         768       
_________________________________________________________________
dropout_84 (Dropout)         (None, 4, 4, 192)         0         
_________________________________________________________________
max_pooling2d_36 (MaxPooling (None, 2, 2, 192)         0         
_________________________________________________________________
dropout_85 (Dropout)         (None, 2, 2, 192)         0         
_________________________________________________________________
separable_conv2d_79 (Separab (None, 1, 1, 96)          19296     
_________________________________________________________________
activation_104 (Activation)  (None, 1, 1, 96)          0         
_________________________________________________________________
batch_normalization_70 (Batc (None, 1, 1, 96)          384       
_________________________________________________________________
dropout_86 (Dropout)         (None, 1, 1, 96)          0         
_________________________________________________________________
separable_conv2d_80 (Separab (None, 1, 1, 10)          1066      
_________________________________________________________________
activation_105 (Activation)  (None, 1, 1, 10)          0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 10)                0         
_________________________________________________________________
activation_106 (Activation)  (None, 10)                0         
=================================================================
Total params: 100,021
Trainable params: 98,485
Non-trainable params: 1,536
```

## 50 epochs (from epoch 43, this model is beating the accuracy..)
```python
Epoch 1/50
390/390 [==============================] - 40s 102ms/step - loss: 1.6298 - acc: 0.4200 - val_loss: 1.3403 - val_acc: 0.5348
Epoch 2/50
390/390 [==============================] - 33s 84ms/step - loss: 1.1516 - acc: 0.5945 - val_loss: 1.1136 - val_acc: 0.6216
Epoch 3/50
390/390 [==============================] - 33s 85ms/step - loss: 0.9844 - acc: 0.6550 - val_loss: 0.9029 - val_acc: 0.6886
Epoch 4/50
390/390 [==============================] - 32s 83ms/step - loss: 0.8891 - acc: 0.6865 - val_loss: 0.9014 - val_acc: 0.6915
Epoch 5/50
390/390 [==============================] - 33s 84ms/step - loss: 0.8271 - acc: 0.7107 - val_loss: 0.8026 - val_acc: 0.7230
Epoch 6/50
390/390 [==============================] - 33s 84ms/step - loss: 0.7783 - acc: 0.7296 - val_loss: 0.8323 - val_acc: 0.7220
Epoch 7/50
390/390 [==============================] - 33s 84ms/step - loss: 0.7338 - acc: 0.7400 - val_loss: 0.7251 - val_acc: 0.7527
Epoch 8/50
390/390 [==============================] - 32s 83ms/step - loss: 0.7058 - acc: 0.7553 - val_loss: 0.7905 - val_acc: 0.7326
Epoch 9/50
390/390 [==============================] - 32s 83ms/step - loss: 0.6792 - acc: 0.7632 - val_loss: 0.7883 - val_acc: 0.7325
Epoch 10/50
390/390 [==============================] - 33s 83ms/step - loss: 0.6526 - acc: 0.7737 - val_loss: 0.6532 - val_acc: 0.7738
Epoch 11/50
390/390 [==============================] - 33s 84ms/step - loss: 0.6321 - acc: 0.7792 - val_loss: 0.6543 - val_acc: 0.7765
Epoch 12/50
390/390 [==============================] - 33s 83ms/step - loss: 0.6164 - acc: 0.7832 - val_loss: 0.6597 - val_acc: 0.7747
Epoch 13/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5968 - acc: 0.7919 - val_loss: 0.7285 - val_acc: 0.7560
Epoch 14/50
390/390 [==============================] - 32s 83ms/step - loss: 0.5862 - acc: 0.7962 - val_loss: 0.6759 - val_acc: 0.7735
Epoch 15/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5748 - acc: 0.7988 - val_loss: 0.6427 - val_acc: 0.7860
Epoch 16/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5581 - acc: 0.8061 - val_loss: 0.6110 - val_acc: 0.7923
Epoch 17/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5439 - acc: 0.8110 - val_loss: 0.6299 - val_acc: 0.7873
Epoch 18/50
390/390 [==============================] - 33s 83ms/step - loss: 0.5374 - acc: 0.8124 - val_loss: 0.6059 - val_acc: 0.8003
Epoch 19/50
390/390 [==============================] - 33s 83ms/step - loss: 0.5291 - acc: 0.8139 - val_loss: 0.5954 - val_acc: 0.8000
Epoch 20/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5162 - acc: 0.8187 - val_loss: 0.5761 - val_acc: 0.8070
Epoch 21/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5138 - acc: 0.8214 - val_loss: 0.6279 - val_acc: 0.7949
Epoch 22/50
390/390 [==============================] - 33s 84ms/step - loss: 0.5028 - acc: 0.8229 - val_loss: 0.6542 - val_acc: 0.7840
Epoch 23/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4945 - acc: 0.8247 - val_loss: 0.5868 - val_acc: 0.8029
Epoch 24/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4856 - acc: 0.8285 - val_loss: 0.5550 - val_acc: 0.8181
Epoch 25/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4792 - acc: 0.8327 - val_loss: 0.5885 - val_acc: 0.8046
Epoch 26/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4707 - acc: 0.8365 - val_loss: 0.5878 - val_acc: 0.8093
Epoch 27/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4652 - acc: 0.8370 - val_loss: 0.5583 - val_acc: 0.8116
Epoch 28/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4655 - acc: 0.8365 - val_loss: 0.5787 - val_acc: 0.8074
Epoch 29/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4557 - acc: 0.8392 - val_loss: 0.6052 - val_acc: 0.8018
Epoch 30/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4483 - acc: 0.8407 - val_loss: 0.5623 - val_acc: 0.8128
Epoch 31/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4413 - acc: 0.8457 - val_loss: 0.5679 - val_acc: 0.8106
Epoch 32/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4409 - acc: 0.8455 - val_loss: 0.5831 - val_acc: 0.8090
Epoch 33/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4351 - acc: 0.8462 - val_loss: 0.5762 - val_acc: 0.8145
Epoch 34/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4364 - acc: 0.8455 - val_loss: 0.5831 - val_acc: 0.8051
Epoch 35/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4259 - acc: 0.8490 - val_loss: 0.5604 - val_acc: 0.8136
Epoch 36/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4278 - acc: 0.8496 - val_loss: 0.5481 - val_acc: 0.8193
Epoch 37/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4232 - acc: 0.8505 - val_loss: 0.5655 - val_acc: 0.8142
Epoch 38/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4156 - acc: 0.8527 - val_loss: 0.5768 - val_acc: 0.8165
Epoch 39/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4096 - acc: 0.8555 - val_loss: 0.5824 - val_acc: 0.8094
Epoch 40/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4050 - acc: 0.8569 - val_loss: 0.6020 - val_acc: 0.8040
Epoch 41/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4029 - acc: 0.8578 - val_loss: 0.5305 - val_acc: 0.8258
Epoch 42/50
390/390 [==============================] - 33s 84ms/step - loss: 0.4014 - acc: 0.8582 - val_loss: 0.5596 - val_acc: 0.8193
Epoch 43/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3959 - acc: 0.8597 - val_loss: 0.5241 - val_acc: 0.8329
Epoch 44/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3975 - acc: 0.8602 - val_loss: 0.5557 - val_acc: 0.8223
Epoch 45/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3878 - acc: 0.8628 - val_loss: 0.5278 - val_acc: 0.8250
Epoch 46/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3911 - acc: 0.8615 - val_loss: 0.5417 - val_acc: 0.8259
Epoch 47/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3821 - acc: 0.8645 - val_loss: 0.5598 - val_acc: 0.8233
Epoch 48/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3821 - acc: 0.8655 - val_loss: 0.5401 - val_acc: 0.8240
Epoch 49/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3766 - acc: 0.8657 - val_loss: 0.5446 - val_acc: 0.8271
Epoch 50/50
390/390 [==============================] - 33s 84ms/step - loss: 0.3765 - acc: 0.8652 - val_loss: 0.5319 - val_acc: 0.8275
Model took 1642.00 seconds to train
```
![image](https://github.com/sridevibonthu/EIP_Session3/blob/master/network1.JPG)

# Network 2 after removing biases
```python
# Define the model - by following vgg 16 architecture
# Define the model - after removing dense layers at the end
model = Sequential()
model.add(SeparableConvolution2D(48, 3, 3, border_mode='same', use_bias= False, input_shape=(32, 32, 3))) #(outputsize - 32, Receptive Field - 3)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(48, 3, 3, use_bias= False)) #(30, 5)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(MaxPooling2D(pool_size=(2, 2))) #(15, 6)
model.add(Dropout(0.2))

model.add(SeparableConvolution2D(96, 3, 3, use_bias= False, border_mode='same')) #(15, 10)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(96, 3, 3, use_bias= False)) #(13, 14)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(MaxPooling2D(pool_size=(2, 2))) #(6, 16)
model.add(Dropout(0.2))

model.add(SeparableConvolution2D(192, 3, 3, use_bias= False, border_mode='same')) #(6, 24)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(192, 3, 3, use_bias= False)) #(4, 32)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(MaxPooling2D(pool_size=(2, 2))) #(2, 36)
model.add(Dropout(0.2))

model.add(SeparableConvolution2D(96, 2, 2, use_bias= False)) #(1, 44)
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))

model.add(SeparableConvolution2D(num_classes, 1, 1, use_bias= False)) #(1, 44)
model.add(Activation('relu'))

model.add(Flatten())
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_9 (Separabl (None, 32, 32, 48)        171       
_________________________________________________________________
activation_10 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_8 (Batch (None, 32, 32, 48)        192       
_________________________________________________________________
dropout_11 (Dropout)         (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_10 (Separab (None, 30, 30, 48)        2736      
_________________________________________________________________
activation_11 (Activation)   (None, 30, 30, 48)        0         
_________________________________________________________________
batch_normalization_9 (Batch (None, 30, 30, 48)        192       
_________________________________________________________________
dropout_12 (Dropout)         (None, 30, 30, 48)        0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 15, 15, 48)        0         
_________________________________________________________________
dropout_13 (Dropout)         (None, 15, 15, 48)        0         
_________________________________________________________________
separable_conv2d_11 (Separab (None, 15, 15, 96)        5040      
_________________________________________________________________
activation_12 (Activation)   (None, 15, 15, 96)        0         
_________________________________________________________________
batch_normalization_10 (Batc (None, 15, 15, 96)        384       
_________________________________________________________________
dropout_14 (Dropout)         (None, 15, 15, 96)        0         
_________________________________________________________________
separable_conv2d_12 (Separab (None, 13, 13, 96)        10080     
_________________________________________________________________
activation_13 (Activation)   (None, 13, 13, 96)        0         
_________________________________________________________________
batch_normalization_11 (Batc (None, 13, 13, 96)        384       
_________________________________________________________________
dropout_15 (Dropout)         (None, 13, 13, 96)        0         
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 6, 6, 96)          0         
_________________________________________________________________
dropout_16 (Dropout)         (None, 6, 6, 96)          0         
_________________________________________________________________
separable_conv2d_13 (Separab (None, 6, 6, 192)         19296     
_________________________________________________________________
activation_14 (Activation)   (None, 6, 6, 192)         0         
_________________________________________________________________
batch_normalization_12 (Batc (None, 6, 6, 192)         768       
_________________________________________________________________
dropout_17 (Dropout)         (None, 6, 6, 192)         0         
_________________________________________________________________
separable_conv2d_14 (Separab (None, 4, 4, 192)         38592     
_________________________________________________________________
activation_15 (Activation)   (None, 4, 4, 192)         0         
_________________________________________________________________
batch_normalization_13 (Batc (None, 4, 4, 192)         768       
_________________________________________________________________
dropout_18 (Dropout)         (None, 4, 4, 192)         0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 2, 2, 192)         0         
_________________________________________________________________
dropout_19 (Dropout)         (None, 2, 2, 192)         0         
_________________________________________________________________
separable_conv2d_15 (Separab (None, 1, 1, 96)          19200     
_________________________________________________________________
activation_16 (Activation)   (None, 1, 1, 96)          0         
_________________________________________________________________
batch_normalization_14 (Batc (None, 1, 1, 96)          384       
_________________________________________________________________
dropout_20 (Dropout)         (None, 1, 1, 96)          0         
_________________________________________________________________
separable_conv2d_16 (Separab (None, 1, 1, 10)          1056      
_________________________________________________________________
activation_17 (Activation)   (None, 1, 1, 10)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_18 (Activation)   (None, 10)                0         
=================================================================
Total params: 99,243
Trainable params: 97,707
Non-trainable params: 1,536
_________________________________


Epoch 1/50
390/390 [==============================] - 18s 46ms/step - loss: 1.6471 - acc: 0.4156 - val_loss: 1.2999 - val_acc: 0.5505
Epoch 2/50
390/390 [==============================] - 16s 41ms/step - loss: 1.2041 - acc: 0.5745 - val_loss: 1.1947 - val_acc: 0.5853
Epoch 3/50
390/390 [==============================] - 16s 41ms/step - loss: 1.0343 - acc: 0.6355 - val_loss: 1.0917 - val_acc: 0.6222
Epoch 4/50
390/390 [==============================] - 16s 41ms/step - loss: 0.9438 - acc: 0.6695 - val_loss: 0.9607 - val_acc: 0.6730
Epoch 5/50
390/390 [==============================] - 16s 41ms/step - loss: 0.8692 - acc: 0.6947 - val_loss: 0.9462 - val_acc: 0.6823
Epoch 6/50
390/390 [==============================] - 16s 41ms/step - loss: 0.8207 - acc: 0.7142 - val_loss: 0.8421 - val_acc: 0.7126
Epoch 7/50
390/390 [==============================] - 16s 41ms/step - loss: 0.7777 - acc: 0.7287 - val_loss: 0.8211 - val_acc: 0.7210
Epoch 8/50
390/390 [==============================] - 16s 41ms/step - loss: 0.7478 - acc: 0.7389 - val_loss: 0.8014 - val_acc: 0.7337
Epoch 9/50
390/390 [==============================] - 16s 41ms/step - loss: 0.7149 - acc: 0.7496 - val_loss: 0.7767 - val_acc: 0.7335
Epoch 10/50
390/390 [==============================] - 16s 41ms/step - loss: 0.6924 - acc: 0.7559 - val_loss: 0.7104 - val_acc: 0.7594
Epoch 11/50
390/390 [==============================] - 16s 41ms/step - loss: 0.6726 - acc: 0.7641 - val_loss: 0.7756 - val_acc: 0.7394
Epoch 12/50
390/390 [==============================] - 16s 41ms/step - loss: 0.6525 - acc: 0.7716 - val_loss: 0.7092 - val_acc: 0.7613
Epoch 13/50
390/390 [==============================] - 16s 41ms/step - loss: 0.6370 - acc: 0.7771 - val_loss: 0.6590 - val_acc: 0.7769
Epoch 14/50
390/390 [==============================] - 16s 41ms/step - loss: 0.6190 - acc: 0.7851 - val_loss: 0.7223 - val_acc: 0.7573
Epoch 15/50
390/390 [==============================] - 16s 41ms/step - loss: 0.6045 - acc: 0.7864 - val_loss: 0.6420 - val_acc: 0.7826
Epoch 16/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5909 - acc: 0.7951 - val_loss: 0.6825 - val_acc: 0.7677
Epoch 17/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5808 - acc: 0.7941 - val_loss: 0.6475 - val_acc: 0.7818
Epoch 18/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5678 - acc: 0.8025 - val_loss: 0.6232 - val_acc: 0.7901
Epoch 19/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5581 - acc: 0.8047 - val_loss: 0.6372 - val_acc: 0.7863
Epoch 20/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5511 - acc: 0.8086 - val_loss: 0.5971 - val_acc: 0.7993
Epoch 21/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5401 - acc: 0.8099 - val_loss: 0.6419 - val_acc: 0.7822
Epoch 22/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5315 - acc: 0.8134 - val_loss: 0.6761 - val_acc: 0.7773
Epoch 23/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5274 - acc: 0.8162 - val_loss: 0.5919 - val_acc: 0.7988
Epoch 24/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5230 - acc: 0.8150 - val_loss: 0.6030 - val_acc: 0.7983
Epoch 25/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5054 - acc: 0.8209 - val_loss: 0.5855 - val_acc: 0.8034
Epoch 26/50
390/390 [==============================] - 16s 41ms/step - loss: 0.5012 - acc: 0.8259 - val_loss: 0.5804 - val_acc: 0.8036
Epoch 27/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4920 - acc: 0.8282 - val_loss: 0.6029 - val_acc: 0.8004
Epoch 28/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4908 - acc: 0.8282 - val_loss: 0.6351 - val_acc: 0.7927
Epoch 29/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4865 - acc: 0.8303 - val_loss: 0.5933 - val_acc: 0.7995
Epoch 30/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4710 - acc: 0.8341 - val_loss: 0.5957 - val_acc: 0.8020
Epoch 31/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4745 - acc: 0.8319 - val_loss: 0.6312 - val_acc: 0.7929
Epoch 32/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4673 - acc: 0.8348 - val_loss: 0.6031 - val_acc: 0.8036
Epoch 33/50
390/390 [==============================] - 16s 40ms/step - loss: 0.4598 - acc: 0.8390 - val_loss: 0.5949 - val_acc: 0.8059
Epoch 34/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4503 - acc: 0.8407 - val_loss: 0.5913 - val_acc: 0.8023
Epoch 35/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4543 - acc: 0.8406 - val_loss: 0.5714 - val_acc: 0.8129
Epoch 36/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4438 - acc: 0.8449 - val_loss: 0.5718 - val_acc: 0.8109
Epoch 37/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4431 - acc: 0.8445 - val_loss: 0.5863 - val_acc: 0.8092
Epoch 38/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4352 - acc: 0.8464 - val_loss: 0.5673 - val_acc: 0.8120
Epoch 39/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4318 - acc: 0.8482 - val_loss: 0.5938 - val_acc: 0.8063
Epoch 40/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4315 - acc: 0.8481 - val_loss: 0.6155 - val_acc: 0.7980
Epoch 41/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4292 - acc: 0.8474 - val_loss: 0.5635 - val_acc: 0.8129
Epoch 42/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4217 - acc: 0.8530 - val_loss: 0.6113 - val_acc: 0.8004
Epoch 43/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4195 - acc: 0.8506 - val_loss: 0.5657 - val_acc: 0.8182
Epoch 44/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4170 - acc: 0.8510 - val_loss: 0.5880 - val_acc: 0.8080
Epoch 45/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4133 - acc: 0.8537 - val_loss: 0.5573 - val_acc: 0.8146
Epoch 46/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4148 - acc: 0.8516 - val_loss: 0.6260 - val_acc: 0.7958
Epoch 47/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4078 - acc: 0.8548 - val_loss: 0.5994 - val_acc: 0.8070
Epoch 48/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4010 - acc: 0.8569 - val_loss: 0.5902 - val_acc: 0.8075
Epoch 49/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4041 - acc: 0.8569 - val_loss: 0.5500 - val_acc: 0.8205
Epoch 50/50
390/390 [==============================] - 16s 41ms/step - loss: 0.4010 - acc: 0.8593 - val_loss: 0.5513 - val_acc: 0.8191
Model took 798.77 seconds to train
```
It crossed 82%
