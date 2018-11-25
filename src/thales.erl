%%% File    : thales.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(thales).
-export([variable/1, test/0]).

-include("node.hrl").

%% User defined variables in an expression.
variable(Name) ->
  PlaceholderNode = placeholder_op:op(),
  PlaceholderNode0 = PlaceholderNode#node{name = Name},
  PlaceholderNode0.

test() ->
  X1 = thales:variable("x1"),
  X2 = thales:variable("x2"),
  X3 = node:mul(X1, X2),
  Y = node:add(X3, X1),
  FeedMap = #{X1=>4,X2=>5},
  executer:run([Y], FeedMap).

%% X1 = thales:variable("x1").
%% X2 = thales:variable("x2").
%% Y = thales:test(X1,X2).
%% FeedMap = #{"x1"=>4,"x2"=>5}.
%% YVal = executer:run([Y], FeedMap).
