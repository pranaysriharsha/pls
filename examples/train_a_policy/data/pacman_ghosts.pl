
action(0)::action(stay); action(1)::action(up); action(2)::action(down); 
action(3)::action(left); action(4)::action(right).

sensor_value(0)::ghost(up).
sensor_value(1)::ghost(down).
sensor_value(2)::ghost(left).
sensor_value(3)::ghost(right).

prev_action(0)::prev_act(stay); prev_action(1)::prev_act(up); prev_action(2)::prev_act(down);
prev_action(3)::prev_act(left); prev_action(4)::prev_act(right).

% transition(Action, NextPos)
transition(stay,here).
transition(left,left).
transition(right,right).
transition(up,up).
transition(down,down).

prev_sensor(0)::prev_ghost(up).
prev_sensor(1)::prev_ghost(down).
prev_sensor(2)::prev_ghost(left).
prev_sensor(3)::prev_ghost(right).

unsafe_next :- action(A), transition(A, NextPos), ghost(NextPos).   

safe_next :- \+ unsafe_next.
