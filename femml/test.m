x1 = 0.5;
x2 = 0;
t0 = 0;
tf = 5;
[time, sol] = ode45(@fun, [t0 tf], [x1; x2]);
plot(time, sol(:, 1))


function xdot = fun(t, x)
    mu = 0.1;
    m = 100;
    k = 1500;
    g = 9.81;
    xdot = [x(2); - (mu * m * g * sign(x(2)) + k * x(1)) / m];
end