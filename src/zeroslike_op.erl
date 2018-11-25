%%% File    : zeroslike_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  25 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(zeroslike_op).

-export([op/1, compute/2, gradient/2]).
-include("node.hrl").

%% add two nodes return a new node.
op(Node0) ->
  Name = "Zeroslike " ++ Node0#node.name,
  Node = #node{name=Name,op=fun zeroslike_op:compute/2,grad=fun zeroslike_op:gradient/2,inputs=[Node0]},
  Node.

%% Returns ones_like of the same shape as input.
compute(_, [Val0]) ->
  lists:duplicate(length(Val0), 0).

%% Given gradient of add node, return gradient contributions to each input.
gradient(Node, _) ->
  [zeroslike_op:op(lists:nth(1,Node#node.inputs))].
