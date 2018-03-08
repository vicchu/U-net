from dataprovider import Dataprovider
import net,shutil,os

if __name__ == '__main__':


    shutil.rmtree('log')
    os.mkdir('log')
    os.mkdir('log/train')
    os.mkdir('log/test')


    model_args = {
        'batch_norm':True,
        'n_class':5,
        'features':16,
        'loss_name':'focal_loss'
    }

    data = Dataprovider(time=5,init_dataset=True)
    model = net.Model(dataprovider=data,**model_args)
    train = net.Train(dataprovider=data, model=model, batch_size=32)
    train.train(100000)