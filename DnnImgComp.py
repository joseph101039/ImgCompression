
from AutoEncoder import *

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_data = mnist.train
valid_data = mnist.validation
test_data = mnist.test


model_1 = Autoencoder( n_features=28*28,
                     learning_rate= 0.0005,
                     #learning_rate= 0.003,    ## ASUS-Joseph-18080601
                     #n_hidden=[512,32,4],
                     n_hidden=[512,32,14],
                     alpha=0.0,
                    )
model_1.fit(X=train_data.images,
           Y=train_data.images,
           epochs=20,  
           #epochs=40,   ## ASUS-Joseph-18080601
           validation_data=(valid_data.images,valid_data.images),
           test_data=(test_data.images,test_data.images),
           batch_size = 8,
          )

fig, axis = plt.subplots(2, 15, figsize=(15, 2))

for i in range(0,15):
    img_original = np.reshape(test_data.images[i],(28,28))
    axis[0][i].imshow(img_original, cmap='gray')
    img = np.reshape(model_1.predict(test_data.images[i]),(28,28))
    axis[1][i].imshow(img, cmap='gray')
plt.show()