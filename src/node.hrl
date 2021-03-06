%%% File    : node.hrl
%%% Author  : Ilhan Adiyaman <ilhanadiyaman@yahoo.com>
%%% Description : Automatic Differentiation
%%% Created :  24 Nov 2018 by Ilhan Adiyaman <ilhanadiyaman@yahoo.com>

-record(node, {name, op, grad, const_attr=0, inputs=[]}).
