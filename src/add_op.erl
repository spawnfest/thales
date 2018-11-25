%%% File    : add_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(add_op).

-export([op/2, compute/2, gradient/2]).
-include("node.hrl").

%% add two nodes return a new node.
op(Node1, Node2) ->
  Name = Node1#node.name ++ "+" ++ Node2#node.name,
  Node = #node{name=Name,op=fun add_op:compute/2,grad=fun add_op:gradient/2,inputs=[Node1, Node2]},
  Node.

%% Given values of input node, return result of element-wise addition.
compute(_, [Val1, Val2]) ->
  lists:zipwith(fun(X, Y) -> X+Y end, Val1, Val2).

%% Given gradient of add node, return gradient contributions to each input.
gradient(_, OutputGrad) ->
  [OutputGrad, OutputGrad].
