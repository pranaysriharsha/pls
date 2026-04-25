% actions
action(0)::action(do_nothing);
action(1)::action(accelerate);
action(2)::action(brake);
action(3)::action(turn_left);
action(4)::action(turn_right).

% states (discretized)
sensor_value(0)::grass(in_front).
sensor_value(1)::grass(on_the_left).
sensor_value(2)::grass(on_the_right).

% previous actions
prev_action(0)::prev_act(do_nothing);
prev_action(1)::prev_act(accelerate);
prev_action(2)::prev_act(brake);
prev_action(3)::prev_act(turn_left);
prev_action(4)::prev_act(turn_right).

% previous sensor states
prev_sensor(0)::prev_grass(in_front).
prev_sensor(1)::prev_grass(on_the_left).
prev_sensor(2)::prev_grass(on_the_right).


% unsafe_1: turning toward the side that has grass (one-sided grass only)
unsafe_1 :- grass(on_the_left), \+ grass(on_the_right), action(turn_left).
unsafe_1 :- grass(on_the_right), \+ grass(on_the_left), action(turn_right).

% unsafe_2: temporal — accelerating into front grass on two consecutive steps
%   i.e., X̄(grass(in_front) ∧ act(accelerate)) ∧ (grass(in_front) ∧ act(accelerate))
unsafe_2 :- grass(in_front), action(accelerate), prev_grass(in_front), prev_act(accelerate).

unsafe_next :- unsafe_1; unsafe_2.

safe_next :- \+ unsafe_next.
safe_action(A) :- action(A).