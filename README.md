![Thales](https://github.com/spawnfest/thales/raw/master/imgs/thales.jpg "Logo")

# Thales

Reverse-mode Automatic Differentiation in pure Erlang.

## Introduction

Automatic differentiation(also known as _algorithmic differentiation_ or _computational differentiation_) is a technique that enables the computation of the partial derivatives of numerical computations with respect to any of their inputs.

Reverse mode of Automatic Differentiation requires a more explicit representation of the computation as a graph, which must be built and then traversed both forwards and backwards to get numerical results for the function and its derivative.

In recent years, automatic differentiation have become an important component of many machine learning frameworks such as [MXNet](https://mxnet.incubator.apache.org), [Torch](http://torch.ch), [Theano](https://github.com/Theano/Theano) and [TensorFlow](https://www.tensorflow.org). It is using for minimization of some differentiable loss function which is the heart of many machine learning models. These frameworks usually use reverse mode Automatic Differentiation since it is more efficient than the forward mode.

## Implementation

The initial implementation is carried out as part of the [Spawnfest 2018](https://spawnfest.github.io) hackathon. Therefore, due to the short implementation time, it is treated as a proof of concept.

Our code should be able to construct simple mathematical expressions, eg. `Y = X1 * X2 + X1` , and evaluate their outputs as well as their gradients, e.g. `Y`, `dY/dX1` and `dY/dX2`.

Key concepts and data structures that we would need to implement are:

- Computation graph and node
- Operators, e.g. Add, Mul, Placeholder, Log
- Construction of gradient nodes given forward graph
- Executor

## How to compile

Use rebar3 to compile as below.

```console
$ rebar3 compile
```

## How to run

Enter the build directory and run `erl`.

```console
$ cd _build/default/lib/thales/ebin
$ erl
Eshell V10.1.2  (abort with ^G)
1> thales:show_me_how().
```

## Quick start

If you don't want to copy paste the things below, just run `thales:show_me_how()`. It will run exactly same thing below and print out `Y_val`, `Grad_X1_Val` and `Grad_X2_Val`.

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

According to reverse-mode automatic differentiation algorithm, we create a gradient node for each node in the existing graph and return those that user are interested in evaluating. Now, the computation graph looks like this:

![Computation Graph](https://github.com/spawnfest/thales/raw/master/imgs/graph.png "Computation Graph")

Now we can feed the values of the inputs and evaluate the gradients using `executer`.

```erlang
X1_Val = [20],
X2_Val = [30],
FeedMap = #{X1=>X1_Val,X2=>X2_Val},
[Y_Val, Grad_X1_Val, Grad_X2_Val] = executer:run([Y, Grad_X1, Grad_X2], FeedMap).
```

`Grad_X1_Val` and `Grad_X2_Val` now contain the values of `dY/dX1` and `dY/dX2`.

## More examples

More examples are placed in `thales.erl` module.

- `thales:test_identity()`: Tests identity gradient.
- `thales:test_add_two_vars()`: Tests two variables' addition.
- `thales:test_add_by_const()`: Tests constant addition.
- `thales:test_mul_by_const()`: Tests constant multiplication.
- `thales:test_mul_two_vars()`: Tests two variables' multiplication.
- `thales:test_add_mul_mix_1()`, `thales:test_add_mul_mix_2()`: Tests different combinations of addition and multiplication.

## How to add new operators

- First, create a module by defining `op/2`, `compute/2` and `gradient/2` functions.
- You can use `add_op.erl` as boilerplate.
- Add operator to the `node.erl`.

## Next steps

- I couldn't setup eunit, due to time limit. So the first next step would be that!
- It is not hard to add more operators such as sin(), cos(), pow(), ln(), sqrt().
- Matrix multiplication operator could be interesting.
- We need to test gradient of gradients.
- File structure can be improved.
- More documentation.
- Packaging.
- Mathematical operation order can be problem while using `node:add()` and `node:mul()`. Operator overloading can be built.

## References

- [Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation)
- [Introduction to Automatic Differentiation](http://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/)
- [Backpropogation is Just Steepest Descent with Automatic Differentiation](https://idontgetoutmuch.wordpress.com/2013/10/13/backpropogation-is-just-steepest-descent-with-automatic-differentiation-2/)
