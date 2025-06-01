from pred_new_sequence import *

if __name__ == '__main__':
    # args = parse_args(sys.argv[1:])

    print('Loading sequences...')
    set_name = 'Test'
    # sequences = load_fasta_sequences(args.seq)
    sequences = load_fasta_sequences(f'data/deep-starr/Sequences_{set_name}.fa')

    print('Loading model...')
    # model = load_model(args.model, PARAMS)
    # model = load_model('models/DeepSTARR.model', PARAMS)
    # model = load_model('models/DeepSTARR_different_adam.model', PARAMS)
    # model = load_keras_model('models/Model_DeepSTARR.h5')

    # seeds = [1234, 2787, 123, 72, 4895, 2137, 18, 4253, 9731]
    # seeds = [7898, 2211, 7530, 9982, 7653, 4949, 3008, 1105, 7]
    seeds = [7899, 7897, 7796, 7697, 4898, 4896, 1238]#, 1235, 1237, 7654, 9876]
    
    for seed in seeds:
        print('Loading model...')
        model = load_model(f'models/deep-starr/DeepSTARR_{seed}.model', PARAMS)

        print('Predicting...')
        pred_dev, pred_hk = predict(model, set_name)  # ta funkcja do zmiany
        out_prediction = pd.DataFrame({'Sequence': sequences, 'Predictions_dev': pred_dev, 'Predictions_hk': pred_hk})
        
        out_filename = f'outputs/deep-starr/Pred_new_torch_{seed}_{set_name}.txt'
        out_prediction.to_csv(out_filename, sep='\t', index=False)
        print(f'\nPredictions saved to {out_filename}\n')