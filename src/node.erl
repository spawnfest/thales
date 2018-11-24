%%% File    : node.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(node).
-export([add/2, mul/2]).

-include("node.hrl").

%% Adding two nodes return a new node.
add(Self, Other) ->
  NewNode = if
    is_number(Other) ->
      add_by_const_op:op(Self, Other);
    true ->
      add_op:op(Self, Other)
  end,
  NewNode.

%% Multiplying two nodes return a new node.
mul(Self, Other) ->
  NewNode = if
    is_number(Other) ->
      mul_by_const_op:op(Self, Other);
    true ->
      mul_op:op(Self, Other)
  end,
  NewNode.
