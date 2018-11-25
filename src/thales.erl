%%% File    : thales.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(thales).
-export([variable/1, gradients/2, test/0, show_me_how/0]).

-include("node.hrl").

%% User defined variables in an expression.
variable(Name) ->
  PlaceholderNode = placeholder_op:op(),
  PlaceholderNode0 = PlaceholderNode#node{name = Name},
  PlaceholderNode0.

% Take gradient of output node with respect to each node in NodeList.
% OutputNode: output node that we are taking derivative of.
% NodeList: list of nodes that we are taking derivative wrt.
% Returns: A list of gradient values, one for each node in NodeList respectively.
gradients(OutputNode, NodeList) ->
  % a map from node to a list of gradient contributions from each output node
  NodeToOutputGradsList = maps:new(),
  % Special note on initializing gradient of OutputNode as oneslike_op(OutputNode):
  % We are really taking a derivative of the scalar reduce_sum(OutputNode)
  % instead of the vector OutputNode. But this is the common case for loss function.
  NodeToOutputGradsList0 = maps:put(OutputNode, [oneslike_op:op(OutputNode)], NodeToOutputGradsList),
  % Traverse graph in reverse topological order given
  % the OutputNode that we are taking gradient wrt.
  ReverseTopoOrder = lists:reverse(helper:find_topo_sort([OutputNode])),
  {NodeToOutputGradResult, _} = lists:foldl(
    fun(Node, {NodeToOutputGradAcc, NodeToOutputGradListAcc}) ->
      % Step 1: sum partial adjoints from output edges
      OutputGradsList = maps:get(Node, NodeToOutputGradListAcc),
      OutputGrad1 = lists:nth(1, OutputGradsList), %% TODO: add more
      OutputGradList0 = lists:sublist(OutputGradsList, 2, 10),
      OutputGrad = case helper:is_empty(OutputGradList0) of
        true -> OutputGrad1;
        false -> lists:foldl(
          fun(OutputGradElement, OutputGradAcc)->
            node:add(OutputGradAcc, OutputGradElement)
          end, OutputGrad1, OutputGradList0)
      end,
      NodeToOutputGrad0 = maps:put(Node, OutputGrad, NodeToOutputGradAcc),
      % Step 2: calc partial adjoints for inputs given node.op and grad
      case Node#node.inputs == [] of
        true ->
          {NodeToOutputGrad0, NodeToOutputGradListAcc};
        false ->
          InputGradList = executer:apply_op(Node#node.grad, Node, OutputGrad),
          {_, NodeToOutputGradListAccResult} = lists:foldl(
            fun(NodeInput, {Index, NodeToOutputGradListAcc0}) ->
              NodeToOutputGradListAcc1 = case maps:is_key(NodeInput, NodeToOutputGradListAcc0) of
                                            false ->
                                              maps:put(NodeInput, [lists:nth(Index, InputGradList)], NodeToOutputGradListAcc0);
                                            true ->
                                              GradI = lists:nth(Index, InputGradList),
                                              GradList = maps:get(NodeInput, NodeToOutputGradListAcc0),
                                              GradList0 = lists:append(GradList, [GradI]),
                                              maps:put(NodeInput, GradList0, NodeToOutputGradListAcc0)
                                          end,
              Index0 = Index + 1,
              {Index0, NodeToOutputGradListAcc1}
            end, {1, NodeToOutputGradListAcc}, Node#node.inputs),
          {NodeToOutputGrad0, NodeToOutputGradListAccResult}
        end
    end, {maps:new(), NodeToOutputGradsList0}, ReverseTopoOrder),
  GradNodeList = lists:foldl(
    fun(Node, GradNodeListAcc) ->
                lists:append(GradNodeListAcc, [maps:get(Node, NodeToOutputGradResult)])
    end, [], NodeList),
  GradNodeList.

show_me_how() ->
  X1 = thales:variable("x1"),
  X2 = thales:variable("x2"),
  X3 = node:mul(X1, X2),
  Y = node:add(X3, X1),
  [Grad_X1, Grad_X2] = thales:gradients(Y, [X1, X2]),
  X1_Val = [20, 20, 20],
  X2_Val = [30, 30, 30],
  FeedMap = #{X1=>X1_Val,X2=>X2_Val},
  executer:run([Y, Grad_X1, Grad_X2], FeedMap).

test() ->
  X1 = thales:variable("x1"),
  X2 = thales:variable("x2"),
  X3 = node:mul(X1, X2),
  Y = node:add(X3, X1),
  FeedMap = #{X1=>[4],X2=>[5]},
  executer:run([Y], FeedMap).


%% X1 = thales:variable("x1").
%% X2 = thales:variable("x2").
%% Y = thales:test(X1,X2).
%% FeedMap = #{"x1"=>[4],"x2"=>[5]}.
%% YVal = executer:run([Y], FeedMap).
