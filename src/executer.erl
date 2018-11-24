%%% File    : executer.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

%% Executor computes values for a given subset of nodes in a computation graph.
-module(executer).

-export([run/2]).
-include("node.hrl").

%% Computes values of nodes in EvalNodeList given computation graph.
%% EvalNodeList: list of nodes whose values need to be computed.
%% FeedMap: list of variable nodes whose values are supplied by user.
%% Returns: A list of values for nodes in EvalNodeList.
run(EvalNodeList, FeedMap) ->
  % Traverse graph in topological sort order and compute values for all nodes.
  TopoOrder = helper:find_topo_sort(EvalNodeList),
  FeedMap0 = lists:foldl(fun(Node, FeedMapAcc)->
                io:fwrite("FMA:~p~n",[FeedMapAcc]),
                io:fwrite("NI:~p~n",[Node#node.inputs]),
                case maps:is_key(Node#node.name, FeedMap) of
                  true -> InputVals = lists:foldl(fun(InputNode, InputValsAcc) ->
                                io:fwrite("MapsGet:~p~n",[maps:get(InputNode#node.name,FeedMapAcc)]),
                                InputValsAcc = lists:append(InputValsAcc, maps:get(InputNode#node.name,FeedMapAcc)),
                                io:fwrite("IVA:~p~n",[InputValsAcc])
                              end, [], Node#node.inputs),
                          io:fwrite("~p~n",[InputVals]),
                          maps:put(Node#node.name, 666, FeedMapAcc);
                  false -> FeedMapAcc
                end
              end, FeedMap, TopoOrder),
  io:fwrite("~p~n",[FeedMap0]).
