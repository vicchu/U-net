from dataprovider import Dataprovider
import net

if __name__ == '__main__':


    model_kw = {
        'batch_norm':True,
        'n_class':5,
        'features':16,
    }


    


    data = Dataprovider(time=5)
    model = net.Model(dataprovider=data,loss_name='focal_loss',model_kw=model_kw)
    train = net.Train(dataprovider=data, model=model, batch_size=32)
    train.train(100000)