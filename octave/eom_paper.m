pkg load control

%% https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2a7d3669a2ce0d9ea7a3b5edbd12c454dd551562

g = 9.81;
L = 0.5;
M_p = 2.0;
M_r = 0.1;
R = 0.05;


%%

X = 1/3 * (M_p*(M_p+6*M_r)*L)/(M_p+(3/2)*M_r);
Y = M_p/((M_p+(3/2)*M_r)*R) + 1/L;

A23 = g*(1- ((4/3) * L*M_p/X));
A43 = g*M_p/X;

B2 = 4*L*Y/(3*X) - 1/(M_p*L);
B4 = -Y/X;

%%

A = [0,1,0,0;
     0,0,A23, 0;
     0,0,0,1;
     0,0,A43,0]

B = [0; B2; 0; B4];

C = [1,0,0,0];
%C = [0,0,1,0];

%%

stname = {'x', 'xdot', 'theta', 'thetadot'};

sys = ss(A, B, C, 0);

Q = [1,0,0,0;
    0,1,0,0;
    0,0,1,0;
    0,0,0,1];
R = 1;

eigA = eig(A)

%%

%[K,S,P] = lqr(sys,Q,R)

[K,S,P] = lqr(A,B,Q,R);

K

eigCL = eig(A-B*K)

sysCL = ss(A-B*K, B)
step(sysCL, 10)



