mb = 2;
mw = 0.1;
mt = mb + mw;
r = 0.05;
l = 0.5;
Iw = 0.5*mw*r*r;
g = 9.81;

tau = 1.0;
f = 0;

x = 0;
z = 0;
phi = 0;
theta = 0;
theta_dot = 0;

q = [x; z; phi; l; theta];
q_dot = [0; 0; 0; 0; 0]

M = [mt, 0, 0, mb*sin(theta), mb*l*cos(theta);
    0, mt, 0, mb*cos(theta), -mb*l*sin(theta);
    0, 0, Iw, 0, 0;
    mb*sin(theta), mb*cos(theta), 0, mb, 0;
    mb*l*cos(theta), -mb*l*sin(theta), 0, 0, mb*l*l];

C = [0, 0, 0, 2*mb*cos(theta)*theta_dot, -mb*l*sin(theta)*theta_dot;
     0, 0, 0, -2*mb*sin(theta)*theta_dot, -mb*l*cos(theta)*theta_dot;
     0, 0, 0, 0, 0;
     0, 0, 0, 0, -mb*l*theta_dot;
     0, 0, 0, 2*mb*l*theta_dot, 0];

G = [0; g*mt; 0; g*mb*cos(theta); -g*mb*l*sin(theta)];

S = transpose(
    [0, 0, 1, 0, -1;
     0, 0, 0, 1, 0]);

Jc = transpose(
    [1, 0, -r, 0, 0;
      0, 1, 0, 0, 0]);

u = [tau; f];



q_dot_dot = inv(M)*(S*u - C*q_dot - G);


syms x z phi l theta x_dot z_dot phi_dot l_dot theta_dot

q = [x; z; phi; l; theta; x_dot; z_dot; phi_dot; l_dot; theta_dot];

M_lin = [
    mt, 0, 0, mb*theta, mb*l;
    0, mt, 0, mb, -mb*l*theta;
    0, 0, Iw, 0, 0;
    mb*theta, mb, 0, mb, 0;
    mb*l, -mb*l*theta, 0, 0, mb*l*l];

C_lin = [
    0, 0, 0, 2*mb*theta_dot, -mb*l*theta*theta_dot;
    0, 0, 0, -2*mb*theta*theta_dot, -mb*l*theta_dot;
    0, 0, 0, 0, 0;
    0, 0, 0, 0, -mb*l*theta_dot;
    0, 0, 0, 2*mb*l*theta_dot, 0];

G_lin = [0; g*mt; 0; g*mb; -g*mb*l*theta];

q_dot_dot = inv(M)*(S*u - C*q_dot - G)
