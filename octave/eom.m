pkg load symbolic;

syms m x_dd M l theta theta_d theta_dd r Tau g;

eqn_0 = m*x_dd + M*x_dd + M*l*sin(theta)*theta_d^2 + M*l*cos(theta)*theta_dd == r*Tau;
eqn_1 = -M*l*x_dd*cos(theta) + M*l^2 *theta_dd -M*g*l*sin(theta) == Tau;

% Isolate theta_dd in eqn 0
theta_dd_isolated = simplify(solve(eqn_0, theta_dd));

% Plug in theta_dd value into eqn 1.
e1 = simplify(subs(eqn_1, theta_dd, theta_dd_isolated));

% Now, e1 is eqn 1 without any thetas. Let's isolate for x_dd
x_dd_isolated = simplify(solve(e1,x_dd))

% Now, let's plug x_dd into theta_dd_isolated to get an equation for theta_dd with no x_dd in it
theta_dd_islated = simplify(subs(theta_dd_isolated, x_dd, x_dd_isolated))

