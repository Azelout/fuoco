from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation, Multiply, Add
from tensorflow.keras.models import Model
from tensorflow.keras.saving import save_model
from utils import load_images, edit_config, get_config

def attention_block(x, g, inter_channel):
    # Attention module
    theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1))(x)
    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1))(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), strides=(1, 1))(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    out = Multiply()([x, sigmoid_xg])
    return out

def attention_unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder with attention gates
    u5 = UpSampling2D((2, 2))(c4)
    attn_5 = attention_block(c3, u5, 256)
    u5 = concatenate([u5, attn_5])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    attn_6 = attention_block(c2, u6, 128)
    u6 = concatenate([u6, attn_6])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    attn_7 = attention_block(c1, u7, 64)
    u7 = concatenate([u7, attn_7])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = attention_unet()
model.summary()

images, masks = load_images(target_size=(128,128)) # On récupère toutes les images

history = model.fit(images, masks, batch_size=8, epochs=20, validation_split=0.2)

save_model(model, f'assets/model/fuoco_0_0_{int(get_config("MODEL_VERSION"))+1}.keras')
#edit_config("MODEL_VERSION", get_config("MODEL_VERSION")+1)
