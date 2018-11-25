%%% File    : helper.erl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-module(helper).
-export([find_topo_sort/1, topo_sort_dfs/3, sum/1]).

-include("node.hrl").

%% Given a list of nodes, return a topological sort list of nodes ending in them.
%% A simple algorithm is to do a post-order DFS traversal on the given nodes,
%% going backwards based on input edges. Since a node is added to the ordering
%% after all its predecessors are traversed due to post-order DFS, we get a topological
%% sort.
find_topo_sort(NodeList) ->
  {_, TopoOrder0} = lists:foldl(fun(Node, {Visited, TopoOrder}) ->
                            topo_sort_dfs(Node, Visited, TopoOrder)
                          end, {sets:new(), []}, NodeList),
  TopoOrder0.


%% Post-order DFS
topo_sort_dfs(Node, Visited, TopoOrder) ->
  case sets:is_element(Node, Visited) of
    true -> NewVisited = Visited;
    false -> NewVisited = sets:add_element(Node, Visited)
  end,
  {Visited0, TopoOrder0} = lists:foldl(fun(N, {VisitedAcc, TopoOrderAcc}) ->
                                    topo_sort_dfs(N, VisitedAcc, TopoOrderAcc)
                                 end, {NewVisited, TopoOrder}, Node#node.inputs),
  TopoOrder1 = lists:append(TopoOrder0, [Node]),
  {Visited0, TopoOrder1}.

%% Sum the element in the list
sum(L) ->
   sum(L, 0).
sum([H|T], Acc) ->
   sum(T, H + Acc);
sum([], Acc) ->
   Acc.
