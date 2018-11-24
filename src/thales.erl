%%% File    : thales.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(thales).
-export([variable/1, test/2]).

-include("node.hrl").

%% User defined variables in an expression.
variable(Name) ->
  PlaceholderNode = placeholder_op:op(),
  PlaceholderNode0 = PlaceholderNode#node{name = Name},
  PlaceholderNode0.

test(Var1, Var2) ->
  Y = node:add(Var1, Var2),
  Y.
