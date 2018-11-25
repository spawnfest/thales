# Thales

Reverse-mode Automatic Differentiation in pure Erlang.

## Introduction

Automatic differentiation(also known as _algorithmic differentiation_ or _computational differentiation_) is a technique that enables the computation of the partial derivatives of numerical computations with respect to any of their inputs.

Reverse mode of Automatic Differentiation requires a more explicit representation of the computation as a graph, which must be built and then traversed both forwards and backwards to get numerical results for the function and its derivative.

In recent years, automatic differentiation have become an important component of many machine learning frameworks such as MXNet, Torch, Theano and TensorFlow. It is using for minimization of some differentiable loss function which is the heart of many machine learning models. These frameworks usually use reverse mode Automatic Differentiation since it is more efficient than the forward mode.

## Implementation

The initial implementation is carried out as part of the [Spawnfest 2018](https://spawnfest.github.io) hackathon. Therefore, due to the short implementation time, it is treated as a proof of concept.

Our code should be able to construct simple mathematical expressions, eg. `Y = X1 * X2 + X1` , and evaluate their outputs as well as their gradients, e.g. `Y, dY/dX1 and dY/dX2`.

Key concepts and data structures that we would need to implement are:

- Computation graph and node
- Operators, e.g. Add, Mul, Placeholder, Log
- Construction of gradient nodes given forward graph
- Executor

## How to use it

Here we'll use a simple example to show the API and data structures of `thales` module.

Suppose our expression is `Y = X1 * X2 + X1`, we first define our variables `X1` and `X2` symbolically.

```erlang
X1 = thales:variable("x1"),
X2 = thales:variable("x2").
```

Then, you can define the expression for `Y`.

```erlang
X3 = node:add(X1, X2),
Y = node:mul(X3, X1).
```

In order to evaluate the gradients of `Y` wrt. `X1` and `X2`, wee need to construct the gradient nodes, `Grad_X1` and `Grad_X2`.

```erlang
[Grad_X1, Grad_X2] = thales:gradients(Y, [X1, X2]).
```

According to reverse-mode automatic differentiation algorithm, we create a gradient node for each node in the existing graph and return those that user are interested in evaluating.

Now we can feed the values of the inputs and evaluate the gradients using `executer`.

```erlang
X1_Val = [2, 2, 2],
X2_Val = [3, 3, 3],
FeedMap = #{X1=>X1_Val,X2=>X2_Val},
[Y_Val, Grad_X1_Val, Grad_X2_Val] = executer:run([Y, Grad_X1, Grad_X2], FeedMap).
```

`Grad_X1_Val` and `Grad_X2_Val` now contain the values of `dY/dX1` and `dY/dX2`.
