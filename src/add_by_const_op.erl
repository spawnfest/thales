%%% File    : add_by_const_op.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(add_by_const_op).

-export([op/2, compute/2, gradient/2]).
-include("node.hrl").

%% element-wise add a node by a constant.
op(Node0, ConstVal) ->
  Name = Node0#node.name ++ "+" ++ integer_to_list(ConstVal),
  Node = #node{name=Name,op=fun add_by_const_op:compute/2,grad=fun add_by_const_op:gradient/2,const_attr=ConstVal,inputs=[Node0]},
  Node.

%% Given values of input node, return result of element-wise addition.
compute(Node, [Val0]) ->
  Val0 + Node#node.const_attr.

%% Given gradient of add node, return gradient contribution to input.
gradient(_, OutputGrad) ->
  [OutputGrad].
