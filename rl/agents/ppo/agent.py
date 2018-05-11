import tensorflow as tf

class PPOAgent():

    def __init__(self, network, network_data_format, ):

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        nminibatches = 4
        nbatch = nenvs*nsteps // nminibatches
        ch = get_input_channels()
        ob_space = {
            'screen'  : [None, res, res, ch['screen']],
            'minimap' : [None, res, res, ch['minimap']],
            'flat'    : [None, ch['flat']],
            'available_actions' : [None, ch['available_actions']]
        }

        step_model  = network(...)
        train_model = network(...)

        def train():
            pass

        def save():
            pass

        def get_global_step():
            return sess.run(global_step)

        self.get_global_step = get_global_step
        self.get_value = step_model.get_value
        self.initial_state = step_model.initial_state
        self.save = save
        self.step = step
        self.train = train


def compute_policy_entropy():
    pass


def compute_policy_log_probs():
    pass
