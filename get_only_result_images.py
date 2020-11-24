import numpy as np 
import matplotlib.pyplot as plt 
    
def get_images(path : str):
    with open(path + 'done_count.txt', 'r') as f:
        done_count_str = f.readline()
    
    done_count = int(done_count_str)
    if done_count == 0:
        return

    epoch_array = np.load(path + 'epoch_array.npy')
    test_acc_for_epoch = np.zeros_like(epoch_array, dtype = np.float)
    train_acc_for_epoch = np.zeros_like(epoch_array, dtype = np.float)

    test_f1_for_epoch = np.zeros_like(epoch_array, dtype = np.float)
    train_f1_for_epoch = np.zeros_like(epoch_array, dtype = np.float)

    for i in range(done_count):
        test_acc = np.load(path + 'test_acc_for_epoch_{}.npy'.format(i))
        train_acc = np.load(path + 'train_acc_for_epoch_{}.npy'.format(i))

        length = len(test_acc)

        test_acc_for_epoch[:length] += test_acc
        train_acc_for_epoch[:length] += train_acc
        
        test_f1 = np.load(path + 'test_f1_for_epoch_{}.npy'.format(i))
        train_f1 = np.load(path + 'train_f1_for_epoch_{}.npy'.format(i))

        length = len(test_f1)

        test_f1_for_epoch[:length] += test_f1
        train_f1_for_epoch[:length] += train_f1

    test_acc_for_epoch /= done_count
    train_acc_for_epoch /= done_count

    test_f1_for_epoch /= done_count
    train_f1_for_epoch /= done_count
    
    np.savez(path + 'num_layers=9_epoch_acc_and_f1', 
        epoch_array = epoch_array, 
        train_f1 = train_f1_for_epoch, test_f1 = test_f1_for_epoch,
        train_acc = train_acc_for_epoch, test_acc = test_acc_for_epoch
    )

    plt.plot(epoch_array[:length], test_f1[:length])
    plt.plot(epoch_array[:length], train_f1[:length])
    plt.legend([
        'test f1',
        'train f1'
    ])
    plt.savefig(path + 'epoch_f1.png', dpi = 300)

    plt.figure()
    plt.plot(epoch_array[:length], test_acc[:length])
    plt.plot(epoch_array[:length], train_acc[:length])
    plt.legend([
        'test acc',
        'train acc'
    ])
    plt.savefig(path + 'epoch_acc.png', dpi = 300)

    plt.show()

if __name__ == '__main__':
    ground_path = 'epoch_k-fold_only\\num_layers=9\\'
    training_path = ground_path + 'training_2\\'
    get_images(training_path)
