%%% File    : add_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(add_op).

-export([op/2, compute/2, gradient/2]).
-include("node.hrl").

%% add two nodes return a new node.
op(Node1, Node2) ->
  Name = io:format("(~p+~p)", [Node1, Node2]),
  Node = #node{name=Name,op="AddOp",inputs={Node1, Node2}},
  io:fwrite("~p~n",[Node#node.op]),
  io:fwrite("~p~n",[Node#node.name]),
  io:fwrite("~p~n",[Node#node.inputs]),
  Node.

%% Given values of input node, return result of element-wise addition.
compute(Node, {Val1, Val2}) ->
  Val1 + Val2.

%% Given gradient of add node, return gradient contributions to each input.
gradient(Node, OutputGrad) ->
  {OutputGrad, OutputGrad}.
