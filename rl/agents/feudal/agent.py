class FeudalAgent():

    def __init__(self, policy, args):

        print('\n### Feudal Agent #######')
        print('######################\n')


        def train(obs, states, actions, returns, returns_intr, adv_m, adv_w, s, goals, summary=False):
            pass


        def save(path, step=None):
            os.makedirs(path, exist_ok=True)
            print("Saving agent to %s, step %d" % (path, sess.run(global_step)))
            ckpt_path = os.path.join(path, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=global_step)


        def get_global_step():
            return sess.run(global_step)


        self.train = train
        self.step = step_model.step
        self.get_value = step_model.get_value
        self.save = save
        self.get_global_step = get_global_step
