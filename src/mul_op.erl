%%% File    : mul_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(mul_op).

-export([op/2, compute/2, gradient/2]).
-include("node.hrl").

%% add two nodes return a new node.
op(Node1, Node2) ->
  Name = Node1#node.name ++ "*" ++ Node2#node.name,
  Node = #node{name=Name,op=fun mul_op:compute/2,grad=fun mul_op:gradient/2,inputs=[Node1, Node2]},
  Node.

%% Given values of input node, return result of element-wise multiplication.
compute(_, [Val1, Val2]) ->
  lists:zipwith(fun(X, Y) -> X*Y end, Val1, Val2).

%% Given gradient of multiply node, return gradient contributions to each input.
gradient(Node, OutputGrad) ->
  [First, Second] = Node#node.inputs,
  [node:mul(OutputGrad, Second), node:mul(OutputGrad, First)].
