% actions
action(0)::action(do_nothing);
action(1)::action(accelerate);
action(2)::action(brake).

% previous actions (mapped from prev_action input array)
prev_action(0)::prev_act(do_nothing);
prev_action(1)::prev_act(accelerate);
prev_action(2)::prev_act(brake).

% states (discretized)
sensor_value(0)::sensor(near_grass);
sensor_value(1)::sensor(on_road).

% previous states (mapped from prev_sensor input array)
prev_sensor(0)::prev_sensor(near_grass);
prev_sensor(1)::prev_sensor(on_road).

% Example Constraint: Include a rule for dangerous sequences
% A crash happens with 0.95 probability if we are near grass, accelerating now, and were accelerating previously.
0.95::crash :- sensor(near_grass), action(accelerate), prev_act(accelerate).

% Safe next state is when there is no crash
safe_next :- \+ crash.

% Safe actions (all valid actions)
safe_action(A) :- action(A).
