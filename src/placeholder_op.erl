%%% File    : placeholder_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(placeholder_op).

-export([op/0, compute/2, gradient/2]).
-include("node.hrl").

%% Op to feed value to a nodes.
op() ->
  Node = #node{op="PlaceholderOp"},
  io:fwrite("~p~n",[Node#node.op]),
  Node.

%% No compute function since node value is fed directly in Executor.
compute(Node, {Val0, Val1}) ->
  ok.

%% No gradient function since node has no inputs.
gradient(Node, OutputGrad) ->
  ok.
