from dataprovider import Dataprovider
import net

if __name__ == '__main__':

    data = Dataprovider(time=5)
    model = net.Model(dataprovider=data,loss_name='focal_loss')
    train = net.Train(dataprovider=data, model=model, batch_size=32)
    train.train(100000)