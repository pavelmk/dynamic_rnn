import paragraph_dynamic_rnn

class ParagraphRNNRegressor:
    def __init__(self):
        ## define hyper-parameters based on empirical search
        params = {}
        
        params['max_seq_length'] = 5000
        params['batch_size'] = 32
        params['n_epochs'] = 15

        ## dynamic LSTM params
        default_num_units = 256
        params['num_units'] = default_num_units
        params['num_layers'] = 2
        params['input_keep_prob'] = 1.0
        params['output_keep_prob'] = 0.525
        params['state_keep_prob'] = 0.3
        # If 0, no final_fc_layer
        params['final_fc_layer_size'] = 0
        params['fc_layer_dropout'] = 0.2

        ## the ridge loss coefficient -- defines relative contribution of ridge loss
        # params['gamma'] = 1.0/default_num_units
        params['gamma'] = 0.25

        ## ADAM optimizer
        params['learning_rate'] = 1e-4
        params['beta1'] = 0.5
        params['beta2'] = 0.9
        params['decay_rate'] = 0.9

        ## i/o directories
        params['log_dir'] = "./artifacts"
        params['graph_save_dir'] = "{}/saved_graph/".format(params['log_dir'])
        params['graph_save_name'] = "trained_paragraph_rnn"

        self.params = params


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """ Trains Dynamic RNN defined in paragraph_dynamicrnn_cmd, producing frozen TensorFlow graph
            artifacts in `artifacts/` directory. For an example of how to process training data that's
            been run through an embedding, please refer to the sample notebook.

            Please see documentation of `paragraph_dynamic_rnn.train_paragraph_rnn()` for variable descriptions.
        """
        paragraph_dynamic_rnn.train_paragraph_rnn(self.params, X_train, y_train, X_test, y_test)


    def predict(self, X):
        """ Predicts labels for input `X` based on previously trained deep network.

            Please see documentation of `paragraph_dynamic_rnn.predict_scores()` for variable descriptions.
        """
        return paragraph_dynamic_rnn.predict_scores(self.params, X)
