pkg load control

m_c = 0.25;
m_p = 3.0;
g = 9.81;
L = 0.25;
d1 = 0.01;
d2 = 0.01;

A = [0,0,1,0;
    0,0,0,1;
    0,g*m_p/m_c, -d1/m_c, -d2/(L*m_c);
    0, g*(m_c+m_p)/(L*m_c), -d1/(L*m_c), -d2*(m_c+m_p)/(L*L*m_c * m_p)];

B = [0 ; 0; 1/m_c; 1/(L*m_c)];

C = [1, 0, 0, 0;
    0, 0, 1, 0];
D = [0; 0];

stname = {'x', 'theta', 'xdot', 'thetadot'};
sys = ss(A, B, 'stname', stname);

Q = [10,0,0,0;
    0,1,0,0;
    0,0,1,0;
    0,0,0,1];
R = 0.1;

eig(A)

[K,S,P] = lqr(sys,Q,R)

eig(A-B*K)

Acls = A-B*K;
cls = ss(Acls, B, 'stname', stname);
step(cls)


