import torch
import numpy as np
from tqdm import tqdm


# https://arxiv.org/pdf/2208.10836.pdf
def efficacy(model, x, y):
    """Return forgetting score (efficacy)."""
    information_target_data = information_score(model, x, y)
    eff = torch.inf if information_target_data == 0 else 1. / information_target_data
    return eff

def efficacy_upper_bound(model, x, y, ce_loss):
    """Return upper bound for forgetting score (efficacy)."""
    for p in model.parameters():
        p.requires_grad = True
    model.zero_grad()
    predictions = model(x)
    loss = ce_loss(predictions, y)
    loss.backward()
    squared_norm = gradient_norm(model) ** 2
    # gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    # gradient = torch.concat([grad.data.flatten() for grad in gradients])
    # gradient_norm = torch.linalg.norm(gradient)
    # squared_norm = gradient_norm ** 2
    return torch.inf if squared_norm == 0 else 1. / squared_norm

def information_score(model, x, y, training=False):
    """
    Compute Fisher-based information score for the given model and data.
    The training argument determines if the resulting tensor requires grad and also if the computational graph should be created for the gradient.
    """
    # get model prediction
    predictions = torch.log_softmax(model(x), dim=-1)

    # if x is just a single data point, expand the tensor by an additional dimension, such that x.shape = [1, n], expand y and predictions accordingly
    y = y if len(x.shape) > 1 else y[None, :]
    # guarantee that y is one-hot encoded
    y = y if len(y.shape) > 1 else torch.tensor(onehot(y, 10))
    predictions = predictions if len(x.shape) > 1 else predictions[None, :]
    x = x if len(x.shape) > 1 else x[None, :]
    num_data_points = x.shape[0]

    information = torch.tensor([0.], requires_grad=training)
    # accumulate information score for all data points
    for i in range(num_data_points):
        model.zero_grad()
        label = torch.argmax(y[i])
        prediction = predictions[i][label]
        # gradient of model prediction w.r.t. the model parameters
        gradient = torch.autograd.grad(prediction, model.parameters(), create_graph=training, retain_graph=True)
        for derivative in gradient:
            information = information + torch.sum(derivative**2)

    # "convert" single-valued tensor to float value
    information = information[0]
    # return averaged information score
    return information / num_data_points

def gradient_norm(model):
    """Compute norm of gradient vector w.r.t. the model parameters."""
    gradient = []
    for p in model.parameters():
        if p.grad is not None:
            gradient.append(p.grad.data.flatten())
    gradient = torch.cat(gradient)
    # gradient = torch.concat([p.grad.data.flatten() for p in model.parameters()])
    norm = torch.linalg.norm(gradient)
    return norm

def onehot(y, num_classes):
    labels = np.zeros([y.shape[0], num_classes])
    for i, label in enumerate(y):
        labels[i][label] = 1
    return labels

def approximate_fisher_information_matrix(model, x, y):
    """Levenberg-Marquart approximation of the Fisher information matrix diagonal."""
    # get model prediction
    predictions = torch.log_softmax(model(x), dim=-1)

    # if x is just a single data point, expand the tensor by an additional dimension, such that x.shape = [1, n], expand y accordingly
    y = y if len(x.shape) > 1 else y[None, :]
    x = x if len(x.shape) > 1 else x[None, :]
    num_data_points = x.shape[0]

    # initialize fisher approximation with 0 for each model parameter
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter))

    epsilon = 10e-8
    # accumulate fisher approximation for all data points
    model.train()
    for i in tqdm(range(num_data_points)):
        label = torch.argmax(y[i])
        prediction = predictions[i][label]
        # gradient of model prediction w.r.t. the model parameters
        gradient = torch.autograd.grad(prediction, model.parameters(), retain_graph=True, create_graph=False)
        for j, derivative in enumerate(gradient):
            # add a small constant epsilon to prevent dividing by 0 later
            fisher_approximation[j] += (derivative + epsilon)**2

    return fisher_approximation

def fisher_forgetting(model, x_remaining, y_remaining, alpha):
    """Perform Fisher forgetting as presented in Golatkar et al. 2020."""
    # approximate Fisher information matrix diagonal
    fisher_approximation = approximate_fisher_information_matrix(model, x_remaining, y_remaining)

    for i, parameter in enumerate(model.parameters()):
        # clamping the approximated fisher values according to the implementation details of Golatkar et al.
        noise = torch.sqrt(alpha / fisher_approximation[i]).clamp(max=1e-3) * torch.empty_like(parameter).normal_(0, 1)
        # increasing the noise of the last layer according to the implementation details of Golatkat et al.
        noise = noise * 10 if parameter.shape[-1] == 10 else noise
        parameter.data = parameter.data + noise

    return model


# from resnetc import resnet20
# net = resnet20().to('cuda')
# x = torch.randn(1, 3, 32, 32).to('cuda')
# y = torch.tensor([0]).to('cuda')
# print(efficacy_upper_bound(net, x, y))

