import matplotlib.pyplot as plt
import pickle
import os

######### IMPORTANT
### I added the next line for my computer to run this file from the terminal 
### to run type: python run_plot.py
os.environ['MKL_THREADING_LAYER'] = 'GNU'
######### 

outdir='output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

N_runs = 3

for k in range(N_runs):  
    print('run: ', k)
    os.system('python main.py --optimizer sgd --learning_rate 1e-5 --output=' + outdir+'/sgd.pkl')
    os.system('python main.py --optimizer momentumsgd --learning_rate 1e-5 --output=' + outdir+'/momentumsgd.pkl')
    os.system('python main.py --optimizer rmsprop --learning_rate 1e-5 --output='+outdir+'/rmsprop.pkl')
    os.system('python main.py --optimizer amsgrad --learning_rate 1e-5 --output='+outdir+'/amsgrad.pkl')
    optimizers = ['sgd', 'momentumsgd', 'rmsprop', 'amsgrad']
    
    # Plots the training losses.
    for optimizer in optimizers:
       data = pickle.load(open(outdir+'/'+ optimizer+".pkl", "rb"))
       plt.plot(data['train_loss'], label=optimizer)
    plt.ylabel('Trainig loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('loss_run_' + str(k)+  '.pdf' )
    plt.show()
    
    # Plots the training accuracies.
    for optimizer in optimizers:
        data = pickle.load(open(outdir+'/'+optimizer+".pkl", "rb"))
        plt.plot(data['train_accuracy'], label=optimizer)
    plt.ylabel('Trainig accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('accuracy_run_' + str(k)+  '.pdf' )    
    plt.show()
