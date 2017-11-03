from theano import tensor as T
import theano
from collections import OrderedDict


def get_from_dict_by_name(dict, name):

    out = [x for x in dict if x.name == name]

    if len(out) < 1:
        raise ValueError("There is no variable with the name {} in the dictionary" % name)

    if len(out) > 1:
        raise ValueError("There are more than one variable with the name {} in the dictionary" % name)

    return out[0]


class stochastic_estimator(object):
    def __init__(self):
        pass

    def bprop(self, prob, samples, loss, known_grads):
        raise NotImplementedError()

class REINFORCEMENT(stochastic_estimator):
    def __init__(self, decay=0.9, lambda_reg1=2e-4, lambda2_reg=2e-5, use_cost_std=True,
                 use_biais_reduction=False, **kwargs):
        super(REINFORCEMENT, self).__init__()

        self.decay = decay
        self.lambda_reg1 = lambda_reg1
        self.lambda2_reg = lambda2_reg
        self.use_cost_std = use_cost_std
        self.use_biais_reduction = use_biais_reduction

    def get_new_updates(self, loss):

        new_updates = {}

        # The step (number of minibatches)
        step = theano.shared(0.0, name="step")
        new_updates[step] = step + 1.0

        # Fix decay
        fix_decay = theano.shared(0.0, name="fix_decay")
        new_updates[fix_decay] = self.decay ** (step + 1.0)

        # The running loss average, for the R baseline
        running_loss = theano.shared(0.05, name="running_loss")
        new_baseline = self.decay * running_loss + (1 - self.decay) * loss.mean()
        new_updates[running_loss] = new_baseline

        # The running cost variance
        cost_var = theano.shared(0.5, name="cost_var")
        new_updates[cost_var] = 1.0
        if self.use_cost_std:  # optimal (default : True)
            cost_var_ave = (loss.mean() - new_baseline) ** 2
            new_cost_var = self.decay * cost_var + (1.0 - self.decay) * cost_var_ave
            # new_cost_var = theano.printing.Print('The var:')(new_cost_var)
            new_updates[cost_var] = new_cost_var

        return new_updates

    def bprop(self, prob, samples, loss, known_grads):

        new_known_grads = OrderedDict()
        new_updates = self.get_new_updates(loss)

        # Getting the theano shared variable
        # The running loss average
        running_loss = get_from_dict_by_name(new_updates, 'running_loss')
        new_baseline = new_updates[running_loss]

        # cost rinning variance
        cost_var = get_from_dict_by_name(new_updates, 'cost_var')
        new_cost_var = new_updates[cost_var]

        # The decay for biais reduction
        fix_decay = get_from_dict_by_name(new_updates, 'fix_decay')
        fix_decay = new_updates[fix_decay]

        # The REINFORCEMENT part
        if prob.ndim == 2:
            prob = prob[:, 0].dimshuffle(0, 'x')  # Make it a colomn variable

        column_loss = loss
        if loss.ndim == 2:
            column_loss = loss.dimshuffle(0, 'x')  # Make it a colomn variable

        # reinforcement learning
        cost_std = T.maximum(T.sqrt(new_cost_var), 1.0)  # Some renormalisation
        if self.use_biais_reduction:
            new_baseline = new_baseline / (1 - fix_decay)  # Applying some biais reduction

        centered_reward = (column_loss - new_baseline) / cost_std
        grad = self.lambda_reg1 * (centered_reward) * \
                (samples / (prob + 1e-8)) + self.lambda2_reg * (T.log(prob + 1e-6) + 1)


        new_known_grads[prob] = grad.astype("float32")
        return new_known_grads, new_updates
