%%% File    : mul_by_const_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(mul_by_const_op).

-export([op/2, compute/2, gradient/2]).
-include("node.hrl").

%% element-wise multiply a node by a constant.
op(Node0, ConstVal) ->
  Name = io:format("(~p+~p)", [Node0, ConstVal]),
  Node = #node{name=Name,op="MulByConstOp",const_attr=ConstVal,inputs=[Node0]},
  io:fwrite("~p~n",[Node#node.op]),
  io:fwrite("~p~n",[Node#node.name]),
  io:fwrite("~p~n",[Node#node.inputs]),
  Node.

%% Given values of input node, return result of element-wise multiplication.
compute(Node, {Val0}) ->
  Val0 * Node#node.const_attr.

%% Given gradient of multiplication node, return gradient contribution to input.
gradient(Node, OutputGrad) ->
  {OutputGrad * Node#node.const_attr}.
