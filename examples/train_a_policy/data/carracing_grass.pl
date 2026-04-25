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


% transition
unsafe1 :- grass(on_the_left), \+ grass(on_the_right), action(turn_left).
unsafe1 :- grass(on_the_left), \+ grass(on_the_right), action(accelerate).
unsafe1 :- \+ grass(on_the_left), grass(on_the_right), action(turn_right).
unsafe1 :- \+ grass(on_the_left), grass(on_the_right), action(accelerate).

0.5 :: unsafe2 :-
prev_grass(on_the_left),
prev_act(turn_left),
grass(on_the_left),
action(turn_left),
\+ grass(on_the_right).

0.5 :: unsafe2 :-
prev_grass(on_the_right),
prev_act(turn_right),
grass(on_the_right),
action(turn_right),
\+ grass(on_the_left).


0.5 :: unsafe2 :-
prev_grass(in_front),
prev_act(accelerate),
grass(in_front),
action(accelerate).

unsafe_next :- unsafe1.
unsafe_next :- unsafe2.

safe_next:- \+unsafe_next.
safe_action(A):- action(A).