from dataprovider import Dataprovider
import net

if __name__ == '__main__':


    model_args = {
        'batch_norm':True,
        'n_class':5,
        'features':16,
    }

    data = Dataprovider(time=5,init_dataset=False)
    model = net.Model(dataprovider=data,loss_name='focal_loss',**model_args)
    train = net.Train(dataprovider=data, model=model, batch_size=32)
    train.train(100000)