import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
import cupy as xp
import datetime
import glob
import numpy as np
import os
import shutil
import tarfile
import urllib.request

from PIL import Image, ImageOps

from model import Generator
from model import Discriminator
from visualize import visualize

# Define constants
N = 100     # Minibatch size
SNAPSHOT_INTERVAL = 10
REAL_LABEL = 1
FAKE_LABEL = 0


def main():

    # (Make directories)
    os.mkdir('dataset/') if not os.path.isdir('dataset') else None
    os.mkdir('train/') if not os.path.isdir('train') else None

    # (Download dataset)
    if not os.path.exists('dataset/faces-in-the-wild.npy'):
        url = 'http://tamaraberg.com/faceDataset/faceData.tar.gz'
        response = urllib.request.urlopen(url)
        with open('dataset/faceData.tar.gz', 'wb') as stream:
            stream.write(response.read())
        with tarfile.open('dataset/faceData.tar.gz', 'r') as stream:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(stream, "dataset/")
        train = []
        for path in glob.iglob('dataset/**/*.ppm', recursive=True):
            x = Image.open(path).convert('RGB')
            x = x.resize((64, 64), Image.ANTIALIAS)
            x = np.asarray(x, dtype='f') / 255.
            x = x.transpose(2, 0, 1)
            x = x.reshape((3, 64, 64))
            train.append(np.asarray(x))
        train = np.asarray(train, dtype='f')
        np.save('dataset/faces-in-the-wild', train)
    os.remove('dataset/faceData.tar.gz') if os.path.exists('dataset/faceData.tar.gz') else None
    shutil.rmtree('dataset/faceData', ignore_errors=True)

    # Create samples.
    train = np.load('dataset/faces-in-the-wild.npy').reshape((-1, 3, 64, 64))
    train = np.random.permutation(train)
    validation_z = xp.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype('f')

    # Create the model
    gen = Generator()
    dis = Discriminator()

    # (Use GPU)
    chainer.cuda.get_device(0).use()
    gen.to_gpu()
    dis.to_gpu()

    # Setup optimizers
    optimizer_gen = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_dis = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    optimizer_gen.setup(gen)
    optimizer_dis.setup(dis)

    # (Change directory)
    os.chdir('train/')
    time = datetime.datetime.today().strftime("%Y-%m-%d %H.%M.%S")
    os.mkdir(time)
    os.chdir(time)

    # (Validate input images)
    visualize(train, 'real.png')

    # Training
    for epoch in range(1000):

        # (Validate generated images)
        if (epoch % SNAPSHOT_INTERVAL == 0):
            os.mkdir('%d' % epoch)
            os.chdir('%d' % epoch)
            fake = chainer.cuda.to_cpu(gen(validation_z).data)
            visualize(fake, 'fake.png')
            chainer.serializers.save_hdf5("gen.h5", gen)
            chainer.serializers.save_hdf5("dis.h5", dis)
            os.chdir('..')

        # (Random shuffle samples)
        train = np.random.permutation(train)

        total_loss_dis = 0.0
        total_loss_gen = 0.0

        for n in range(0, len(train), N):

            ############################
            # (1) Update D network
            ###########################
            real = xp.array(train[n:n + N])
            z = xp.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype('f')
            fake = gen(z)
            y_real = dis(real)
            y_fake = dis(fake)
            loss_dis = (F.sum((y_real - REAL_LABEL) ** 2) + F.sum((y_fake - FAKE_LABEL) ** 2)) / np.prod(y_fake.shape)
            dis.cleargrads()
            loss_dis.backward()
            optimizer_dis.update()

            ###########################
            # (2) Update G network
            ###########################
            z = xp.random.uniform(low=-1.0, high=1.0, size=(100, 100)).astype('f')
            fake = gen(z)
            y_fake = dis(fake)
            loss_gen = F.sum((y_fake - REAL_LABEL) ** 2) / np.prod(y_fake.shape)
            gen.cleargrads()
            loss_gen.backward()
            optimizer_gen.update()

            total_loss_dis += loss_dis.data
            total_loss_gen += loss_gen.data

        # (View loss)
        total_loss_dis /= len(train) / N
        total_loss_gen /= len(train) / N
        print(epoch, total_loss_dis, total_loss_gen)


if __name__ == '__main__':
    main()
