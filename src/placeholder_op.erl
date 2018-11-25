%%% File    : placeholder_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(placeholder_op).

-export([op/0, compute/2, gradient/2]).
-include("node.hrl").

%% Op to feed value to a nodes.
op() ->
  Node = #node{op=fun placeholder_op:compute/2},
  Node.

%% No compute function since node value is fed directly in Executor.
compute(Node, Vals) ->
  ok.

%% No gradient function since node has no inputs.
gradient(Node, OutputGrad) ->
  ok.
