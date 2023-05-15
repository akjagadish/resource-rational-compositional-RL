import math
import torch
import gpytorch
import numpy as np
from utils import to_string
from GP import GP, CompositionalMean
from train import MLE
from Kernels import DiagonalKernel, ChangePointKernel
from Kernels import Kernels

class ChoiceModel():
    def __init__(self, training_iters = 50):
        self.training_iters = training_iters
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel)
        self.has_episodic_dict = False
    def new_task(self, *args):
        '''Default is that the model resets prior on new tasks'''
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel)

    def condition(self, X, y):
        self.posterior_gp.set_train_data(X, y, strict=False)

    def predict(self, test_x):
        self._posterior = self.posterior_gp.posterior(test_x)
        self._test_x = test_x
        self._y_preds = self._posterior.mean
        return self._posterior

    def predict_rewards(self, test_x, compute_sd=True):
        posterior = self.predict(test_x)
        mean =posterior.mean.detach().numpy()
        sd = posterior.stddev.detach().numpy() if compute_sd else None
        return mean, sd

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters)

    def update(self, X, y):
        pass

    def transfer(self, *args):
        pass

    def set_kernel(self, linfirst):
        pass


    @staticmethod
    def UCB(posterior, beta):
        mu = posterior.mean
        sigma = posterior.stddev
        upper = mu + (beta * sigma)
        return upper.detach().numpy()

    @staticmethod
    def greedy(posterior):
        return posterior.mean.detach().numpy()

    @staticmethod
    def softmax_sample(arms, Q_vals):
        e_Q = torch.exp(Q_vals)
        p = e_Q / e_Q.sum()
        p_numpy = p.detach().numpy()
        arm = np.random.choice(arms, p =p_numpy)
        return arm

    @staticmethod
    def greedy_choice(arms, Q_vals):
        return arms[torch.argmax(Q_vals)]

    @staticmethod
    def eps_greedy_choice(arms, Q_vals, eps=0.3):
        rand_arm = torch.randint(0, high=len(arms), size=(1,))[0]
        return arms[rand_arm] if torch.rand(1)<eps else arms[torch.argmax(Q_vals)]

    @staticmethod
    def UCB_softmax_choice(arms, means, uncertainities, beta, tau):
        Q_vals = torch.tensor(beta*means + uncertainities*tau)
        p_numpy = torch.nn.functional.softmax(Q_vals).detach().numpy()
        arm = np.random.choice(arms, p=p_numpy)
        return arm

    @staticmethod
    def sticky_UCB_softmax_choice(arms, means, uncertainities, beta, tau, sticky, prev_arm):
        
        Q_vals = torch.tensor(beta*means + uncertainities*tau + sticky*np.eye(len(arms))[prev_arm])
        p_numpy = torch.nn.functional.softmax(Q_vals).detach().numpy()
        arm = np.random.choice(arms, p=p_numpy)
        return arm
    

class GrammarModel(ChoiceModel):
    '''Grammar model only transferring concrete functional knowledge'''
    def __init__(self, grammar, episodic_dict, value_function, choice_function = None, training_iters=50):
        super().__init__()
        self.grammar = grammar
        self.episodic_dict = episodic_dict
        self.value_function = value_function
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), gpytorch.kernels.RBFKernel())
        self.training_iters = training_iters
        self.choice_function = choice_function
        self.has_episodic_dict = True

        self.grammar.initialize()

    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), gpytorch.kernels.RBFKernel()) # new zero mean gp for the first trial

    def transfer(self, feature_vector, rule=None, linfirst=True):
        output = self.episodic_dict.function_transfer(feature_vector) # check if context is similar to anything
        # if it is:
        if type(output) != type(None):
            gps, probabilities = output  # unpack
            mean_module = CompositionalMean(gps, probabilities, rule, linfirst)

            self.posterior_gp.register_mean(mean_module)
        # if it's not, don't update the mean function



    def fit(self, X, y):
        self.grammar.fit(X, y, self.training_iters)  # this procedure finds a MAP kernel
        self.posterior_gp = self.grammar.map_gp  # create a new GP model with this kernel, and the best hyperparam settings

    def update(self, X, y):
        # here we add the latest GP to the EpisodicDictionary

        # update with the final data points gained from the last action
        self.condition(X, y)
        self.episodic_dict.append(self.feature_string, X, y, self.posterior_gp, self.grammar.map_kernel)


class SimpleGrammarModel(GrammarModel):
    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([gpytorch.kernels.ScaleKernel(Kernels.Linear), gpytorch.kernels.ScaleKernel(Kernels.Periodic), gpytorch.kernels.ScaleKernel(Kernels.Periodic + Kernels.Linear)])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)

class SimpleGrammarModelConstrained(GrammarModel):
    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([gpytorch.kernels.ScaleKernel(Kernels.Linear), gpytorch.kernels.ScaleKernel(Kernels.Periodic), gpytorch.kernels.ScaleKernel(Kernels.Periodic + Kernels.Linear)])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)
        self.lin = gpytorch.kernels.ScaleKernel(Kernels.Linear)
        self.per = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(active_dims=0))#gpytorch.kernels.ScaleKernel(Kernels.Periodic)
        self.comp = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(active_dims=0) + Kernels.Linear)
        
    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)

        ## check compositional
        if feature_vector.sum() > 1:
            self.kernel = self.comp

        elif feature_vector[0] == 1:
            self.kernel = self.lin

        else:
            self.kernel = self.per
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel) # new zero mean gp for the first trial

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
        self.grammar.map_kernel = self.kernel


class SimpleGrammarModelConstrainedChangePoint(GrammarModel):
    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50, lin_first = True):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([gpytorch.kernels.ScaleKernel(Kernels.Linear), gpytorch.kernels.ScaleKernel(Kernels.Periodic), gpytorch.kernels.ScaleKernel(Kernels.Periodic + Kernels.Linear)])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)
        self.lin = gpytorch.kernels.ScaleKernel(Kernels.Linear)
        self.per = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(active_dims=0))#gpytorch.kernels.ScaleKernel(Kernels.Periodic)
        self.set_kernel(lin_first)
        
    def set_kernel(self, lin_first):
        if lin_first:
        #self.comp = Kernels.ChangePointKernel(Kernels.Linear, gpytorch.kernels.PeriodicKernel(active_dims=0))  gpytorch.kernels.PeriodicKernel(active_dims=0)
            self.comp = gpytorch.kernels.ScaleKernel(ChangePointKernel(Kernels.Linear, gpytorch.kernels.PeriodicKernel(active_dims=0), t=0., active_dims=0),active_dims=0) #gpytorch.kernels.ScaleKernel(,active_dims=0)
        else:
            self.comp = gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.PeriodicKernel(active_dims=0), Kernels.Linear, t=0., active_dims=0),active_dims=0) #gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.PeriodicKernel(active_dims=0), Kernels.Linear, t=0.),active_dims=0)


    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)

        ## check compositional
        if feature_vector.sum() > 1:
            self.kernel = self.comp

        elif feature_vector[0] == 1:
            self.kernel = self.lin

        else:
            self.kernel = self.per
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel) # new zero mean gp for the first trial

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
        self.grammar.map_kernel = self.kernel

class CompositeModel(ChoiceModel):
    def __init__(self, rule='add', training_iters = 50):
        super().__init__(training_iters)
        self.rule = rule
        self.lin = gpytorch.kernels.ScaleKernel(Kernels.Linear)
        self.per = gpytorch.kernels.ScaleKernel(Kernels.Periodic)
        self.comp = self.lin + self.per if rule == 'add' else NotImplementedError('test')
        self.kernel = self.lin #self.comp

class MeanTracker(ChoiceModel):
    def __init__(self, training_iters = 50):
        super().__init__(training_iters)

        self.kernel = DiagonalKernel()
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel)

class MeanTrackerCompositional(GrammarModel):

    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([DiagonalKernel(), DiagonalKernel(), DiagonalKernel()])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)
        self.lin = DiagonalKernel()
        self.per = DiagonalKernel()
        self.comp = gpytorch.kernels.ScaleKernel(self.lin) +gpytorch.kernels.ScaleKernel(self.per)
        
    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)

        ## check compositional
        if feature_vector.sum() > 1:
            self.kernel = self.comp

        elif feature_vector[0] == 1:
            self.kernel = self.lin

        else:
            self.kernel = self.per
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel) # new zero mean gp for the first trial

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
        self.grammar.map_kernel = self.kernel


class MeanTrackerCompositionalChangePoint(GrammarModel):
    
    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50, lin_first = True):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([DiagonalKernel(), DiagonalKernel(), DiagonalKernel()])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)
        self.lin = DiagonalKernel()
        self.per = DiagonalKernel()
        self.set_kernel(lin_first)

    def set_kernel(self, lin_first):
        if lin_first:
        #self.comp = Kernels.ChangePointKernel(Kernels.Linear, gpytorch.kernels.PeriodicKernel(active_dims=0))  gpytorch.kernels.PeriodicKernel(active_dims=0)
            self.comp = gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.ScaleKernel(self.lin), gpytorch.kernels.ScaleKernel(self.per), t=0., active_dims=0), active_dims=0) #gpytorch.kernels.ScaleKernel(,active_dims=0)
        else:
            self.comp = gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.ScaleKernel(self.per), gpytorch.kernels.ScaleKernel(self.lin), t=0., active_dims=0), active_dims=0) #gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.PeriodicKernel(active_dims=0), Kernels.Linear, t=0.),active_dims=0)

    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)

        ## check compositional
        if feature_vector.sum() > 1:
            self.kernel = self.comp

        elif feature_vector[0] == 1:
            self.kernel = self.lin

        else:
            self.kernel = self.per
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel) # new zero mean gp for the first trial

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
        self.grammar.map_kernel = self.kernel

class RBFModelReset(ChoiceModel):
    def __init__(self, training_iters = 50):
        super().__init__(training_iters)

class RBFModelResetCompositional(GrammarModel):

    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([self.kernel, self.kernel, self.kernel])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)
        self.lin = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.per = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.comp = gpytorch.kernels.ScaleKernel(self.lin) + gpytorch.kernels.ScaleKernel(self.per)
        
    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)

        ## check compositional
        if feature_vector.sum() > 1:
            self.kernel = self.comp

        elif feature_vector[0] == 1:
            self.kernel = self.lin

        else:
            self.kernel = self.per
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel) # new zero mean gp for the first trial

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
        self.grammar.map_kernel = self.kernel

class RBFModelResetCompositionalChangePoint(GrammarModel):
    
    def __init__(self, grammar, episodic_dict, value_function, choice_function=None, training_iters=50, lin_first = True):
        super().__init__(grammar, episodic_dict, value_function, choice_function, training_iters)
        self.grammar.kernels = np.array([self.kernel, self.kernel, self.kernel])
        self.grammar.probabilities = np.ones(len(self.grammar.kernels)) / len(self.grammar.kernels)
        self.lin = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.per = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.set_kernel(lin_first)

    def set_kernel(self, lin_first):
        if lin_first:
        #self.comp = Kernels.ChangePointKernel(Kernels.Linear, gpytorch.kernels.PeriodicKernel(active_dims=0))  gpytorch.kernels.PeriodicKernel(active_dims=0)
            self.comp = gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.ScaleKernel(self.lin), gpytorch.kernels.ScaleKernel(self.per), t=0., active_dims=0), active_dims=0) #gpytorch.kernels.ScaleKernel(,active_dims=0)
        else:
            self.comp = gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.ScaleKernel(self.per), gpytorch.kernels.ScaleKernel(self.lin), t=0., active_dims=0), active_dims=0) #gpytorch.kernels.ScaleKernel(ChangePointKernel(gpytorch.kernels.PeriodicKernel(active_dims=0), Kernels.Linear, t=0.),active_dims=0)

    def new_task(self, feature_vector):
        self.feature_string = to_string(feature_vector)
        self.episodic_dict.new_entry(self.feature_string, feature_vector, self.posterior_gp)

        ## check compositional
        if feature_vector.sum() > 1:
            self.kernel = self.comp

        elif feature_vector[0] == 1:
            self.kernel = self.lin

        else:
            self.kernel = self.per
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel) # new zero mean gp for the first trial

    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
        self.grammar.map_kernel = self.kernel

# NEW CLASS

class RBFModelMemory(ChoiceModel):
    def __init__(self, sees_context_features, training_iters = 50):
        super().__init__(training_iters)
        self.sees_context_features = sees_context_features



    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        self.posterior_gp, self._ml= MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=False)




class RBFModelMemorySubTask(ChoiceModel):
    def __init__(self, training_iters = 50):
        super().__init__(training_iters)

        self.X = None
        self.y = None

    def update(self, X_task, y_task):
        # in this method we add the data points observed at task t to the
        # global dataset used to condition the gp on for future tasks
        if type(self.X) == type(None):
            self.X = X_task
            self.y = y_task
        else:
            self.X = torch.cat((self.X, X_task))
            self.y = torch.cat((self.y, y_task))

    def new_task(self, *args):
        pass

    def reset(self):

        self.X = None
        self.y = None
        self.kernel = gpytorch.kernels.RBFKernel()
        self.posterior_gp = GP(None, None, gpytorch.likelihoods.GaussianLikelihood(), self.kernel)

    def condition(self, X, y):
        # Here we condition the GP on the global dataset and the datapoints
        # observed in the current task
        # If there's no global data set yet, just use the data points from the task, whatever these are
        if type(self.X) == type(None):
            X_ = X
            y_ = y
        # if there is a global dataset, and the current observations are just None, condition on the global dataset
        elif type(X) == type(None):
            X_ = self.X
            y_ = self.y
        # if there is a global dataset and current observations arent none, then concatenate them together
        else: #type(X) != type(None):
            X_ = torch.cat((self.X, X))
            y_ = torch.cat((self.y, y))

        # condition
        self.posterior_gp.set_train_data(X_, y_, strict=False)


    def fit(self, X, y):
        l = gpytorch.likelihoods.GaussianLikelihood()
        gp = GP(X, y, l, self.kernel)
        # make the return kernel argument True, so we can update our kernel hyperparameters
        # as the task progresses
        self.posterior_gp, self._ml, self.kernel = MLE(gp, X, y, learning_rate=0.01, training_iterations=self.training_iters, return_kernel=True)
