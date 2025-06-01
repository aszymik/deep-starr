from train import *

if __name__ == '__main__':
    # seeds = [1234, 2787, 123, 72, 4895, 2137, 18, 4253, 9731]
    # seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    seeds = [7899, 7897, 7796, 7697, 4898, 4896, 1238, 1235, 1237, 7654, 9876]

    train_loader = prepare_input('Train', PARAMS['batch_size'])
    val_loader = prepare_input('Val', PARAMS['batch_size'])
    
    model = DeepSTARR(PARAMS)
    for seed in seeds:
        log_file = f'train_logs/deep-starr/training_log_{seed}.csv'
        trained_model = train(model, train_loader, val_loader, PARAMS, log_file, seed)
        torch.save(model.state_dict(), f'models/deep-starr/DeepSTARR_{seed}.model')