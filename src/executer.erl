%%% File    : executer.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

%% Executor computes values for a given subset of nodes in a computation graph.
-module(executer).

-export([run/2, apply_op/3]).
-include("node.hrl").

%% Computes values of nodes in EvalNodeList given computation graph.
%% EvalNodeList: list of nodes whose values need to be computed.
%% FeedMap: list of variable nodes whose values are supplied by user.
%% Returns: A list of values for nodes in EvalNodeList.
run(EvalNodeList, FeedMap) ->
  % Traverse graph in topological sort order and compute values for all nodes.
  TopoOrder = helper:find_topo_sort(EvalNodeList),
  FeedMap0 = lists:foldl(fun(Node, FeedMapAcc)->
                case maps:is_key(Node#node.name, FeedMap) of
                  false -> InputVals = lists:foldl(fun(InputNode, InputValsAcc) ->
                                                      lists:append(InputValsAcc, [maps:get(InputNode,FeedMapAcc)])
                                                    end, [], Node#node.inputs),
                          Value = executer:apply_op(Node#node.op, Node, InputVals),
                          if
                            is_list(Value) ->
                              maps:put(Node, Value, FeedMapAcc);
                            true ->
                              FeedMapAcc
                          end;
                  true -> FeedMapAcc
                end
              end, FeedMap, TopoOrder),
  NodeValResults = lists:foldl(fun(Node, NodeValResultsAcc) ->
                lists:append(NodeValResultsAcc, [maps:get(Node, FeedMap0)])
  end, [], EvalNodeList),
  NodeValResults.

%% Invokes the operation
apply_op(Op, Node, InputVals) ->
  Op(Node, InputVals).
