import theano
from theano import tensor
from theano.ifelse import ifelse
import numpy as np
import theano.sandbox.rng_mrg as rng_mrg
import mixer
from core.operators import GumbelSoftmax, concreteDist
import collections


BIG=10

class Planner(object):
    def __init__(self, prefix, options, create_param=True, repeat_actions=False,
                 plan_steps=10, ntimesteps=10,
                 inter_size=64, dec_dim=500, batch_size=None, context_dim=-1,
                 use_gate=True, always_recommit=False,
                 bounded_sigm_temp_act=False,
                 do_commit=True,
                 do_layerNorm=False):

        self.repeat_actions = repeat_actions
        self.ntimesteps = ntimesteps
        self.prefix = prefix
        self.inter_size = inter_size
        self.bounded_sigm_temp_act = bounded_sigm_temp_act
        self.dec_dim = dec_dim
        self.context_dim = context_dim
        self.use_gate = use_gate
        self.always_recommit = always_recommit
        self.do_commit = do_commit

        if not "st_estimator" in options:
            options['st_estimator'] = "GumbelSoftmax"
            self.st_estimator = "GumbelSoftmax"

        self.st_estimator = options['st_estimator']

        if self.st_estimator is None:
            self.st_estimator = "GumbelSoftmax"
            options['st_estimator'] = self.st_estimator

        self.rng = rng_mrg.MRG_RandomStreams(seed=1993)

        if 'plan_step' in options:
            self.plan_steps = options['plan_step']
        else:
            self.plan_steps = plan_steps

        self.only_use_w = False
        if 'only_use_w' in options:
            self.only_use_w = options['only_use_w']
            if self.only_use_w:
               print "We will only use the h2 state for the attention."
            else:
                print "We will use all the hidden state for the attention."

        if 'use_gate' in options:  # Shitty way to do it, but it's a pain to add everything everywhere
            self.use_gate = options['use_gate']
            if self.use_gate:
                print "We are using a gate in the planner"
            else:
                print "We won't be using the gate for the planner"

        self.learn_t = False
        if 'learn_t' in options:
            self.learn_t = options['learn_t']
            if self.learn_t:
                print "We are learning the temperature"
            else:
                print "We won't be learning the temperature"

        if self.st_estimator == "REINFORCE":
            print "Using REINFORCE"
        elif self.st_estimator == "GumbelSoftmax":
            print "Using GumbelSoftmax"
        else:
            raise ValueError("Wrong st estimator: {}".format(self.st_estimator))


        self.action_plan_steps = plan_steps
        if 'repeat_actions' in options:
            self.repeat_actions = options['repeat_actions']
            if self.repeat_actions:
               print "We will repeat the action until recommitment (and won't be using gates."
               self.action_plan_steps = 1
               self.use_gate = False
            else:
                print "We We will plan ahead all futur alignment."

        self.do_layerNorm = do_layerNorm
        if 'planning_do_layerNorm' in options:
            self.do_layerNorm = options['planning_do_layerNorm']
            if self.do_layerNorm:
                print "We are doing layernorm in the PAG network"
            else:
                print "We are not doing layernorm in the PAG network"



        self.actionPlanner = ActionPlan(inter_size=inter_size, context_size=context_dim, dec_size=dec_dim,
                                        create_param=create_param, batch_size=batch_size,
                                        repeat_actions=self.repeat_actions,
                                        plan_steps=self.plan_steps, ntimesteps=ntimesteps, options=options)

        if do_commit:
            self.commitplan = CommitmentPlan(create_param=create_param,
                                             bellow_size=dec_dim,
                                             plan_steps=self.plan_steps,
                                             bounded_sigm_temp_act=self.bounded_sigm_temp_act,
                                             options=options, rng=self.rng)
        else:
            print "WARNING, we are not doing any commitment."
            self.commitplan = CommitmentPlan(create_param=create_param,
                                             bellow_size=dec_dim,
                                             plan_steps=self.plan_steps,
                                             bounded_sigm_temp_act=self.bounded_sigm_temp_act,
                                             options=options, rng=self.rng)

        if create_param:
            self.init_params()

    def init_params(self):

        # Action plan -> inter space
        self.inter_w = mixer.norm_weight(self.action_plan_steps, self.inter_size)
        self.inter_b = np.zeros((self.inter_size,)).astype('float32')

        #self.gate_B
        self.gate_B_w = tensor.zeros((1,))
        self.gate_B_c = tensor.zeros((1,))
        self.gate_C = tensor.zeros((1,))

        if self.use_gate:

            if self.only_use_w:
                self.gate_B_w = mixer.norm_weight(self.dec_dim, self.action_plan_steps)

            if not self.only_use_w:
                self.gate_B_c = mixer.norm_weight(self.dec_dim, self.action_plan_steps)

            self.gate_C = mixer.norm_weight(self.context_dim, self.action_plan_steps)

        self.gate_b1 = None
        self.gate_s1 = None

        if self.do_layerNorm:

            self.gate_norm_b1 = np.zeros((self.action_plan_steps,)).astype('float32')
            self.gate_norm_s1 = np.ones((self.action_plan_steps,)).astype('float32')
            
            self.inter_norm_b1 = np.zeros((self.inter_size,)).astype('float32')
            self.inter_norm_s1 = np.ones((self.inter_size,)).astype('float32')

    def compute_gate(self, previous_state_w, previous_state_c, context):


        if self.only_use_w:
            gate_state = tensor.dot(previous_state_w, self.gate_B_w)

        if not self.only_use_w:
            gate_state = tensor.dot(previous_state_c, self.gate_B_c)
        
        # Layer Norm
        gate_state = mixer.ln(gate_state, self.gate_norm_b1, self.gate_norm_s1, self.do_layerNorm)

        gate = gate_state[None, :, :] + tensor.dot(context, self.gate_C)

        gate = tensor.nnet.sigmoid(gate)
        return gate

    def compute_inter(self, action_plan):

        if self.repeat_actions:
            # If we repeat the alttention, we don't compute an intermidiate state
            inter_state = action_plan
        else:

            #LayerNorm
            inter_state = tensor.dot(action_plan, self.inter_w) + self.inter_b
            inter_shape = inter_state.shape
            inter_state = tensor.reshape(inter_state,
                                         (inter_shape[0]*inter_shape[1],
                                          inter_shape[2]))
            
            inter_state = mixer.ln(inter_state, self.inter_norm_b1, self.inter_norm_s1, self.do_layerNorm)
            inter_state = tensor.reshape(inter_state, inter_shape)
            inter_state = tensor.tanh(inter_state)
            
        return inter_state

    def getAlpha(self, previous_state_w, previous_state_c, context, action_plan_m1, commit_plan_tm1, probs_tm1,
                 logits_commit_plan_tm1, commit_origin,
                 probs_origin, params = None):

        if self.only_use_w:
            # If we don't use the caracter state. Just making sure we don't use it.
            previous_state_c = None
        else:
            previous_state_w = None

        if params is not None:
            # If we have a list of parameters (scan is strict for exemple)
            self.setParams(params)

        return self.alpha_compute_all(previous_state_w, previous_state_c, context, action_plan_m1, commit_plan_tm1,
                                      probs_tm1, logits_commit_plan_tm1,
                                          commit_origin, probs_origin)

    def alpha_compute_all(self, previous_state_w, previous_state_c, context,
                          action_plan_m1, commit_plan_tm1, probs_tm1, logits_commit_plan_tm1,
                          commit_origin,
                          probs_origin,
                          ):

        commit_origin.name = 'commit_origin'
        probs_origin.name = 'probs_origin'
        commit_plan_tm1.name = 'commit_plan_tm1'

        inter = self.compute_inter(action_plan_m1)

        # Optimization: did a matrix multiplication to only return the first column.
        first_column_matrix = np.array([1.] + [0.]*(self.plan_steps-1)).astype('float32')
        gt = tensor.dot(commit_plan_tm1, first_column_matrix)

        # Our commitment
        temp_t = 0
        if self.do_commit:
            logits_commit_plan = self.commitplan.new_plan(previous_state_w, previous_state_c)
            logits_commit_plan.name = "logits"
            logits_commit_plan_shift = self._time_shift(logits_commit_plan_tm1)

            #set the last one to a large netagive number.
            logits_commit_plan_shift = tensor.set_subtensor(logits_commit_plan_shift[:, -1], -BIG)



            probs, temp_t = self.commitplan.apply_softmax(logits_commit_plan, previous_state_w,
                                                  previous_state_c)
            probs_shifted, temp_t = self.commitplan.apply_softmax(logits_commit_plan_shift,
                                                          previous_state_w, previous_state_c)
            tmp_gt = gt.dimshuffle((0, 'x'))

            commit_plan = self.commitplan.sample(probs)
            commit_plan_shifted = self.commitplan.sample(probs_shifted)
            commit_plan = tmp_gt * commit_plan + (1 - tmp_gt) * commit_plan_shifted
            logits_commit_plan = tmp_gt * logits_commit_plan + (1 - tmp_gt) * logits_commit_plan_shift

            commit_origin = (1. - tmp_gt) * commit_origin + tmp_gt * commit_plan
            probs_origin = (1 - tmp_gt) * probs_origin + tmp_gt * probs
        else:
            commit_plan = commit_plan_tm1
            probs = commit_plan
            logits_commit_plan = logits_commit_plan_tm1

        gt = tensor.dot(commit_plan, first_column_matrix)

        # New action plan
        # From our current intermediate state and the context
        new_plan = self.actionPlanner.compute_new_plan(context, inter, previous_state_w, previous_state_c)

        # Our alpha
        tmp_gt = gt.dimshuffle(('x', 0, 'x'))

        if not self.repeat_actions:
            if self.do_commit and self.use_gate:
                # Compute the forget get.
                gate = self.compute_gate(previous_state_w, previous_state_c, context)

                # Little trick to avoid playing with theano.if. technically add some dependencies, but...
                action_plan = tmp_gt * ((1. - gate) * new_plan + gate * action_plan_m1) + (1. - tmp_gt) * self._time_shift(
                    action_plan_m1)
            elif self.do_commit:
                action_plan = new_plan * tmp_gt + self._time_shift(action_plan_m1)
            else :
                # For testing purposes
                action_plan = new_plan

            action_plan.name = "action_plan_t"
            alpha = tensor.dot(action_plan, first_column_matrix)

        else:
            # We repeat the last action
            tmp_gt = gt.dimshuffle((0, 'x'))
            action_plan = tmp_gt * new_plan + (1. - tmp_gt) * action_plan_m1
            action_plan.name = "action_plan"

            action_plan = action_plan[:, :, 0].dimshuffle((0, 1, 'x'))
            action_plan.name = "action_plan"

            alpha = action_plan[:, :, 0] #Take the first and only column
            alpha.name = 'alpha'

        return probs, commit_plan, logits_commit_plan, commit_origin, probs_origin, alpha, action_plan, \
                temp_t,

    def getParams(self, return_all = False):

        def _p(pp, name):
            return '%s_%s' % (pp, name)

        params = collections.OrderedDict()

        # Inter representation
        if not self.repeat_actions:
            params[_p(self.prefix, "inter_w")] = self.inter_w
            params[_p(self.prefix, "inter_b")] = self.inter_b

        # Gate
        if self.use_gate or return_all:
            
            #params[_p(self.prefix, "gate_B_b")] = self.gate_B_b

            if self.only_use_w:
                params[_p(self.prefix, "gate_B_w")] = self.gate_B_w

            if not self.only_use_w:
                params[_p(self.prefix, "gate_B_c")] = self.gate_B_c

            params[_p(self.prefix, "gate_C")] = self.gate_C


        # Commit plan
        if return_all:
            params[_p(self.prefix, "commit_ww")] = tensor.zeros((1,))
            params[_p(self.prefix, "commit_wc")] = tensor.zeros((1,))
            params[_p(self.prefix, "commit_b")] = tensor.zeros((1,))
            params[_p(self.prefix, "commit_b")] = tensor.zeros((1,))
            params[_p(self.prefix, "temperature_dec_ww")] = tensor.zeros((1,))
            params[_p(self.prefix, "temperature_dec_wc")] = tensor.zeros((1,))
            params[_p(self.prefix, "temperature_b")] = tensor.zeros((1,))

        if self.do_commit:

            if self.only_use_w:
                params[_p(self.prefix, "commit_ww")] = self.commitplan.commit_ww

            if not self.only_use_w or return_all:
                params[_p(self.prefix, "commit_wc")] = self.commitplan.commit_wc

            params[_p(self.prefix, "commit_b")] = self.commitplan.commit_b

            if self.learn_t or return_all:
                
                if self.only_use_w:
                    params[_p(self.prefix, "temperature_dec_ww")] = self.commitplan.temperature_dec_ww

                if not self.only_use_w or return_all:
                    params[_p(self.prefix, "temperature_dec_wc")] = self.commitplan.temperature_dec_wc

                params[_p(self.prefix, "temperature_b")] = self.commitplan.temperature_b

        # Action plan
        if not self.repeat_actions:
            params[_p(self.prefix, "plan_wi_h")] = self.actionPlanner.plan_wi_h
            params[_p(self.prefix, "plan_bi_h")] = self.actionPlanner.plan_bi_h
        
        if self.only_use_w:
            params[_p(self.prefix, "plan_wd_h_w")] = self.actionPlanner.plan_wd_h_w

        if not self.only_use_w or return_all:
            params[_p(self.prefix, "plan_wd_h_c")] = self.actionPlanner.plan_wd_h_c

        if self.do_layerNorm:# or return_all:
            params[_p(self.prefix, "plan_norm_b1")] = self.actionPlanner.plan_norm_b1
            params[_p(self.prefix, "plan_norm_s1")] = self.actionPlanner.plan_norm_s1
            
            params[_p(self.prefix, "gate_norm_b1")] = self.gate_norm_b1
            params[_p(self.prefix, "gate_norm_s1")] = self.gate_norm_s1


            params[_p(self.prefix, "commit_norm_b1")] = self.commitplan.commit_norm_b1
            params[_p(self.prefix, "commit_norm_s1")] = self.commitplan.commit_norm_s1
            
            params[_p(self.prefix, "inter_norm_b1")] = self.inter_norm_b1
            params[_p(self.prefix, "inter_norm_s1")] = self.inter_norm_s1

        params[_p(self.prefix, "plan_bd_h")] = self.actionPlanner.plan_bd_h
        params[_p(self.prefix, "plan_w_p")] = self.actionPlanner.plan_w_p
        params[_p(self.prefix, "plan_b_p")] = self.actionPlanner.plan_b_p

        return params

    def setParams(self, params):

        def _p(pp, name):
            return '%s_%s' % (pp, name)

        self.inter_w = tensor.zeros((1,))
        self.inter_b = tensor.zeros((1,))

        if not self.repeat_actions:
            self.inter_w = params[_p(self.prefix, "inter_w")]
            self.inter_b = params[_p(self.prefix, "inter_b")]

        # Gate
        self.gate_B = tensor.zeros((1,))
        self.gate_B_w = tensor.zeros((1,))
        self.gate_B_c = tensor.zeros((1,))

        self.gate_C = tensor.zeros((1,))
        if self.use_gate:

            if self.only_use_w:
                self.gate_B_w = params[_p(self.prefix, "gate_B_w")]

            if not self.only_use_w:
                self.gate_B_c = params[_p(self.prefix, "gate_B_c")]

            self.gate_C = params[_p(self.prefix, "gate_C")]

        # Commitment plan
        if self.do_commit:
            
            self.commitplan.commit_ww = tensor.zeros((1,))
            
            if self.only_use_w:
                self.commitplan.commit_ww = params[_p(self.prefix, "commit_ww")]

            self.commitplan.commit_wc = tensor.zeros((1,))
            if not self.only_use_w:
                self.commitplan.commit_wc = params[_p(self.prefix, "commit_wc")]

            self.commitplan.commit_b = params[_p(self.prefix, "commit_b")]

            self.commitplan.temperature_dec_ww = tensor.zeros((1,))
            self.commitplan.temperature_dec_wc = tensor.zeros((1,))

            self.commitplan.temperature_b = tensor.zeros((1,))

            if self.learn_t:
                
                if self.only_use_w:
                    self.commitplan.temperature_dec_ww = params[_p(self.prefix, "temperature_dec_ww")]

                if not self.only_use_w:
                    self.commitplan.temperature_dec_wc = params[_p(self.prefix, "temperature_dec_wc")]

                self.commitplan.temperature_b = params[_p(self.prefix, "temperature_b")]

        # Action plan
        if not self.repeat_actions:
            self.actionPlanner.plan_wi_h = params[_p(self.prefix, "plan_wi_h")]
            self.actionPlanner.plan_bi_h = params[_p(self.prefix, "plan_bi_h")]


        self.actionPlanner.plan_wd_h_c = tensor.zeros((1,))
        
        if self.only_use_w:
            self.actionPlanner.plan_wd_h_w = params[_p(self.prefix, "plan_wd_h_w")]

        self.actionPlanner.plan_wd_h_c = tensor.zeros((1,))
        if not self.only_use_w:
            self.actionPlanner.plan_wd_h_c = params[_p(self.prefix, "plan_wd_h_c")]



        if self.do_layerNorm:
            self.actionPlanner.plan_norm_b1 = params[_p(self.prefix, "plan_norm_b1")]
            self.actionPlanner.plan_norm_s1 = params[_p(self.prefix, "plan_norm_s1")]
            
            self.gate_norm_b1 = params[_p(self.prefix, "gate_norm_b1")]
            self.gate_norm_s1 = params[_p(self.prefix, "gate_norm_s1")]
            self.inter_norm_b1 = params[_p(self.prefix, "inter_norm_b1")]
            self.inter_norm_s1 = params[_p(self.prefix, "inter_norm_s1")]

            self.commitplan.commit_norm_b1 = params[_p(self.prefix, "commit_norm_b1")]
            self.commitplan.commit_norm_s1 = params[_p(self.prefix, "commit_norm_s1")]
        else:
            self.actionPlanner.plan_norm_b1 = tensor.zeros((1,))
            self.actionPlanner.plan_norm_s1 = tensor.zeros((1,))

            self.gate_norm_b1 = tensor.zeros((1,))
            self.gate_norm_s1 = tensor.zeros((1,))
            self.inter_norm_b1 = tensor.zeros((1,))
            self.inter_norm_s1 = tensor.zeros((1,))

            self.commitplan.commit_norm_b1 = tensor.zeros((1,))
            self.commitplan.commit_norm_s1 = tensor.zeros((1,))

        self.actionPlanner.plan_bd_h = params[_p(self.prefix, "plan_bd_h")]
        self.actionPlanner.plan_w_p = params[_p(self.prefix, "plan_w_p")]
        self.actionPlanner.plan_b_p = params[_p(self.prefix, "plan_b_p")]

    def _time_shift(self, input):

        # First roll
        ndim = input.ndim

        shift_matrix = np.identity(self.plan_steps)
        shift_matrix = np.roll(shift_matrix, shift=-1, axis=-1)
        shift_matrix[0] = 0
        shift_matrix = shift_matrix.astype('float32')

        # The shift matrix is for example
        # [[0, 0, 0]
        #  [1, 0, 0]
        #  [0, 1, 0]]
        # for predicting 3 timesteps.
        return tensor.dot(input, shift_matrix)


class CommitmentPlan(object):


    def __init__(self, options, rng, create_param=True,
                 bounded_sigm_temp_act=False,
                 bellow_size=None, plan_steps=5, do_layerNorm=False):

        self.bounded_sigm_temp_act = bounded_sigm_temp_act
        if not "st_estimator" in options:
            options['st_estimator'] = "GumbelSoftmax"
            self.st_estimator = "GumbelSoftmax"


        self.st_estimator = options['st_estimator']
        self.rng = rng
        self.plan_steps=plan_steps

        # Learn the temperature for the gumbel softmax estimator
        self.learn_t = False
        if 'learn_t' in options:
            self.learn_t = options['learn_t']
            if self.learn_t:
                print "We are learning the temperature"
            else:
                print "We won't be learning the temperature"


        self.only_use_w = False
        if 'only_use_w' in options:
            self.only_use_w = options['only_use_w']

        self.do_layerNorm = do_layerNorm
        if 'planning_do_layerNorm' in options:
            self.do_layerNorm = options['planning_do_layerNorm']


        if create_param:
            self.plan_steps = plan_steps
            
            if self.only_use_w:
                self.commit_ww = mixer.norm_weight(bellow_size, plan_steps)

            if not self.only_use_w:
                self.commit_wc = mixer.norm_weight(bellow_size, plan_steps)

            self.commit_b = np.zeros((plan_steps,)).astype('float32')

            self.temperature_dec_ww = None
            self.temperature_dec_wc = None
            self.temperature_b = None

            if self.learn_t:
                #self.temperature_cont_w = mixer.norm_weight(self.context_dim, 1)
                
                if self.only_use_w:
                    self.temperature_dec_ww = mixer.norm_weight(bellow_size, 1)

                if not self.only_use_w:
                    self.temperature_dec_wc = mixer.norm_weight(bellow_size, 1)

                self.temperature_b = (np.zeros((1,)) - 1.8).astype("float32")

            self.commit_norm_s1 = None
            self.commit_norm_b1 = None

            if self.do_layerNorm:

                self.commit_norm_b1 = np.zeros((plan_steps,)).astype('float32')
                self.commit_norm_s1 = np.ones((plan_steps,)).astype('float32')

    def get_temperature(self, previous_state_w, previous_state_c):


        if self.learn_t:
            # Layer Norm
            temperature = self.temperature_b

            if self.only_use_w:
                temperature += tensor.dot(previous_state_w, self.temperature_dec_ww)

            if not self.only_use_w:
                temperature += tensor.dot(previous_state_c, self.temperature_dec_wc)

            # Making sure it's bigger than 1.
            #temperature = tensor.nnet.softplus(temperature) + 1. + 1e-3
            if self.bounded_sigm_temp_act:
                temperature = 1.0 + 4.0*tensor.nnet.sigmoid(temperature)
            else:
                temperature = tensor.nnet.softplus(temperature) + 1. + 1e-3

            return temperature[:, 0].dimshuffle((0, 'x')) # return a matrix, but I want a vector

        else:
            return tensor.ones((previous_state_w.shape[0],))#.dimshuffle((0, 'x'))

    def new_plan(self, previous_state_w, previous_state_c):


        #Layer Norm
        new_commit = self.commit_b
        if self.only_use_w:
            new_commit += tensor.dot(previous_state_w, self.commit_ww)

        if not self.only_use_w:
            new_commit += tensor.dot(previous_state_c, self.commit_wc)
        
        if self.do_layerNorm:
            new_commit = mixer.ln(new_commit, self.commit_norm_b1, self.commit_norm_s1, self.do_layerNorm)
        
        return new_commit

    def apply_softmax(self, new_commit, previous_state_w, previous_state_c):
        temperature = tensor.zeros((1,)) + 1.6
        if self.st_estimator == 'REINFORCE':
            probs = tensor.nnet.softmax(new_commit)
        else:
            if self.learn_t:
                temperature = self.get_temperature(previous_state_w, previous_state_c)
            probs = GumbelSoftmax(temperature)(new_commit, self.rng)

        return probs.astype('float32'), temperature

    def sample(self, probs):

        if self.st_estimator == "REINFORCE":
            samples = self.rng.multinomial(pvals=probs)
        else:
            samples = concreteDist(probs)

        return samples.astype('float32')


class ActionPlan(object):
    def __init__(self, options, context_size, inter_size, dec_size, create_param=True, batch_size=None,
                 repeat_actions=False, plan_steps=5, ntimesteps=10,
                 do_layerNorm=False):

        self.repeat_actions = repeat_actions
        self.plan_steps = plan_steps
        self.ntimesteps = ntimesteps
        self.batch_size = batch_size
        self.context_size = context_size
        self.inter_size = inter_size
        self.dec_size = dec_size
        self.do_layerNorm = do_layerNorm

        self.only_use_w = False
        if 'only_use_w' in options:
            self.only_use_w = options['only_use_w']

        if 'planning_do_layerNorm' in options:
            self.do_layerNorm = options['planning_do_layerNorm']

        if batch_size is not None:
            self.__create_action_plan()

        #If repeat_actions is true, Our plan is a 1x|X| plan
        if self.repeat_actions:
            self.plan_steps = 1


        if create_param == True:
            if not self.repeat_actions:
                self.plan_wi_h = mixer.norm_weight(inter_size, context_size)
                self.plan_bi_h = np.zeros((context_size,)).astype('float32')

            if self.only_use_w:
                self.plan_wd_h_w = mixer.norm_weight(dec_size, context_size)

            if not self.only_use_w:
                self.plan_wd_h_c = mixer.norm_weight(dec_size, context_size)
            self.plan_bd_h = np.zeros((context_size,)).astype('float32')

            self.plan_w_p = mixer.norm_weight(context_size, plan_steps)
            self.plan_b_p = np.zeros((plan_steps,)).astype('float32')

            self.plan_norm_b1 = np.zeros((context_size,)).astype('float32')
            self.plan_norm_s1 = np.ones((context_size,)).astype('float32')

    def __create_action_plan(self):
        self.A = tensor.zeros((self.ntimesteps, self.batch_size, self.plan_steps), dtype='float32')

    def compute_new_plan(self, context, inter, previous_state_w, previous_state_c):

        # Context
        # Done outside to save computation

        # Interstate
        # If we repeat, we ignore the inter state.
        if not self.repeat_actions:
            context_state = context + tensor.dot(inter, self.plan_wi_h) + self.plan_bi_h
        else:
            context_state = context
        # decoder


        
        if self.only_use_w:
            decoder_state = tensor.dot(previous_state_w, self.plan_wd_h_w)

        if not self.only_use_w:
            decoder_state = tensor.dot(previous_state_c, self.plan_wd_h_c)

        decoder_state += self.plan_bd_h

        #Layer Norm
        decoder_state = mixer.ln(decoder_state, self.plan_norm_b1,
                             self.plan_norm_s1, self.do_layerNorm)

        new_plan_ctx = context_state + decoder_state[None, :, :]

        # Non-linearity and blah
        new_plan_h = tensor.tanh(new_plan_ctx)
        new_plan = tensor.dot(new_plan_h, self.plan_w_p) + self.plan_b_p

        return new_plan
