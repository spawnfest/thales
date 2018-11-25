%%% File    : oneslike_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(oneslike_op).

-export([op/1, compute/2, gradient/2]).
-include("node.hrl").

%% add two nodes return a new node.
op(Node0) ->
  Name = "Oneslike " ++ Node0#node.name,
  Node = #node{name=Name,op=fun oneslike_op:compute/2,grad=fun oneslike_op:gradient/2,inputs=[Node0]},
  Node.

%% Returns ones_like of the same shape as input.
compute(_, [Val0]) ->
  lists:duplicate(length(Val0), 1).

%% Given gradient of add node, return gradient contributions to each input.
gradient(Node, OutputGrad) ->
  [OutputGrad, OutputGrad].
