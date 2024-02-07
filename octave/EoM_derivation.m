pkg load symbolic

syms m x_dd M l theta theta_d theta_dd r Tau g

eqn_0 = m*x_dd + M*x_dd + M*l*sin(theta)*theta_d^2 + M*l*cos(theta)*theta_dd == r*Tau
eqn_1 = -M*l*x_dd*cos(theta) + M*l^2 *theta_dd -M*g*l*sin(theta) == Tau

theta_dd_isolated = isolate(eqn_0, theta_dd)
x_dd_isolated = isolate(eqn_1, x_dd)

