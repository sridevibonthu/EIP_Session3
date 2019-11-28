# EIP_Session3
Assignment 3
# Base network accuracy

After running the base network for 50 epochs the validation accuracy is - 82.66

![image](https://github.com/sridevibonthu/EIP_Session3/blob/master/basenwaccuracy.JPG)

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
