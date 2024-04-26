clear; clc; close all
% 输入数据
L = 5e-5; % 长度
Ce0 = 17300; % mol/m^3
M0 = 72; % g/mol
Mn0 = 120000; % g/mol
m = 4; n = 0.5;
Xmax = 0.655; px = 0.004; Vc = 4.19e-24; na = 6.02e23;
alpha = 28; beta = 2; omega = 17300;
k1 = 3.3e-4; % /week
k2 = 0.0058; % (m^3/mol)^0.5/week
D_poly = 1e-9; % m^-2/week
D_pore = 10^4*D_poly;
Col0 = 0; 
Rs0 = 0;
Xc0 = 0.565;
Nchain0 = 10.4;
Nt = 1000000;
t = linspace(0, 60, Nt);
dt = t(2) - t(1);
Nz = 50;
z = linspace(0, L, Nz);
dz = z(2) - z(1);
recordInterval = 10000;
count = 1;
% initial_dt = 7e-12;
% final_dt = 7e-7;
% log_growth_limit = 100 * recordInterval;
RsA = zeros(Nz,1);
ColA = zeros(Nz,1);
RolA = zeros(Nz,1);
CeA = ones(Nz,1)*Ce0;
XcA = ones(Nz,1)*Xc0;
Rs_new = zeros(floor(Nt / recordInterval), Nz);
Col_new = zeros(floor(Nt / recordInterval), Nz);
Rol_new = zeros(floor(Nt / recordInterval), Nz);
Ce_new = zeros(floor(Nt / recordInterval), Nz);
Xc_new = zeros(floor(Nt / recordInterval), Nz);
%%%%%%%%%%%%%%
dRsdt_diff_mat = zeros(floor(Nt / recordInterval), Nz);
Xc_diff_mat = zeros(floor(Nt / recordInterval), Nz);
dRoldt_diff_mat = ones(floor(Nt / recordInterval), Nz)*(k1*Ce0+k2*Ce0*(0/(1-Xc0))^n);
dRsdt_diff = zeros(Nz,1);
Xc_diff = zeros(Nz,1);
dRoldt_diff = zeros(Nz,1);
%%%%%%%%%%%%%%
S = zeros(floor(Nt / recordInterval), Nz);
drolb1 = load('dRoldt_bias-input_layer.txt');
drolb2 = load('dRoldt_bias-first_layer.txt');
drolb4 = load('dRoldt_bias-third_layer.txt');
drolb3 = load('dRoldt_bias-second_layer.txt');
drolw1 = load('dRoldt_weights-input_layer.txt');
drolw2 = load('dRoldt_weights-first_layer.txt');
drolw3 = load('dRoldt_weights-second_layer.txt');
drolw4 = load('dRoldt_weights-third_layer.txt');

bdrolw1 = zeros(size(drolw1));
bdrolw2 = zeros(size(drolw2));
bdrolw3 = zeros(size(drolw3));
bdrolw4 = zeros(size(drolw4));
bdrolb1 = zeros(size(drolb1));
bdrolb2 = zeros(size(drolb2));
bdrolb3 = zeros(size(drolb3));
bdrolb4 = zeros(size(drolb4));

ML = load('ML.txt');
time2 = [0,5,10,15,20,25,30,35,40,50,60];
point2 = [0,0.21,0.24,0.46,0.68,1.47,2.45,4.39,6.52,12.87,17.5];
time_interval = linspace(0, 60, 100);
ml_interpolated = interp1(time_interval, ML, time2, 'linear');
loss = point2-ml_interpolated;
% loss = sum((y - ML(:)).^2) / numel(y);
[droldta4_all, droldta3_all, droldta2_all, droldta1_all] = calculatedrol(drolw1, drolb1, drolw2, drolb2, drolw3, drolb3, drolw4, drolb4, Rs_new, Ce_new, dRsdt_diff_mat, Nz, Nt, recordInterval);
%%
for i = 1:Nt
% 保存当前RsA到新数组的一行
if mod(i-1, recordInterval) == 0 || i == 1
Rs_new(floor((i-1)/recordInterval)+1, :) = RsA';
Col_new(floor((i-1)/recordInterval)+1, :) = ColA';
Rol_new(floor((i-1)/recordInterval)+1,:) = RolA';
Ce_new(floor((i-1)/recordInterval)+1,:) = CeA';
Xc_new(floor((i-1)/recordInterval)+1,:) = XcA';
    dRsdt_diff_mat(floor((i-1)/recordInterval)+1,:) = dRsdt_diff';
    Xc_diff_mat(floor((i-1)/recordInterval)+1,:) = Xc_diff';
    dRoldt_diff_mat(floor((i-1)/recordInterval)+1,:) = dRoldt_diff';
count = count + 1;
end
for j = 2:Nz-1
    dRsdt = k1*CeA(j)+k2*CeA(j)*(ColA(j)/(1-XcA(j)))^n;
dRsdt_diff(j) = dRsdt;
V = alpha*(RsA(j)/Ce0)^beta-(ColA(j)-Col0)/Ce0;
D = D_poly + (1.3*V^2-0.3*V^3)*(D_pore-D_poly);
XcA(j) = Xmax-(Xmax-Xc0)*exp(-px*na*Vc*RsA(j));

    dRoldtann = [RsA(j)/36.483,CeA(j)/17300,dRsdt/203.486];
    droldtz1 = dRoldtann*drolw1+drolb1';
    droldta1 = max(0.01*droldtz1,droldtz1);
    droldtz2 = droldta1*drolw2+drolb2';
    droldta2 = 1./(1+exp(-droldtz2));
    droldtz3 = droldta2*drolw3+drolb3';
    droldta3 = max(0.01*droldtz3,droldtz3);
    droldtz4 = droldta3*drolw4+drolb4;
    dRoldt = real(droldtz4*22.2);

CeA(j) = Ce0-RolA(j)-omega*(XcA(j)-Xc0);
dCol2dz = D*(ColA(j+1)-2*ColA(j)+ColA(j-1))/dz/dz;
dColdt = real(dRoldt + dCol2dz); 
RsA(j) = real(RsA(j) + dRsdt * dt);
ColA(j) = real(ColA(j) + dColdt * dt);
RolA(j) = real(RolA(j) + dRoldt * dt);
% Backpropagation
for backprop_iter = 1:100
for x = 1:Nt/recordInterval
    for y = 1:length(loss)
dLoss_dRol = 2 * (loss(y)); % Assuming a simple mean squared error loss

dLoss_dW4 = droldta3_all{x, j}' * dLoss_dRol;
dLoss_dB4 = sum(dLoss_dRol, 1);
dLoss_dZ3 = dLoss_dRol * drolw4';
dLoss_dA3 = dLoss_dZ3;  % 由于没有激活函数，直接等于对应的梯度

dLoss_dW3 = droldta2_all{x, j}' * dLoss_dA3;
dLoss_dB3 = sum(dLoss_dA3, 1);
dLoss_dZ2 = dLoss_dA3 * drolw3';
dLoss_dA2 = dLoss_dZ2 .* (droldta2_all{x, j} .* (1 - droldta2_all{x, j}));

dLoss_dW2 = droldta1_all{x, j}' * dLoss_dA2;
dLoss_dB2 = sum(dLoss_dA2, 1);
dLoss_dZ1 = dLoss_dA2 * drolw2';
dLoss_dA1 = dLoss_dZ1 .* leaky_relu_derivative(droldta1_all{x, j}, 0.01);

dLoss_dW1 = [Rs_new(x,j)/36.483,Ce_new(x,j)/17300,dRsdt_diff_mat(x,j)/203.486]' * dLoss_dA1;
dLoss_dB1 = sum(dLoss_dA1, 1);

% Accumulate the gradients for each step
bdrolw1 = bdrolw1 + dLoss_dW1;
bdrolw2 = bdrolw2 + dLoss_dW2;
bdrolw3 = bdrolw3 + dLoss_dW3;
bdrolw4 = bdrolw4 + dLoss_dW4;
bdrolb1 = bdrolb1 + dLoss_dB1';
bdrolb2 = bdrolb2 + dLoss_dB2';
bdrolb3 = bdrolb3 + dLoss_dB3';
bdrolb4 = bdrolb4 + dLoss_dB4';
%Update the weights and biases
learning_rate = 0.0001;
backdrolw1 = drolw1 - learning_rate * bdrolw1;
backdrolw2 = drolw2 - learning_rate * bdrolw2;
backdrolw3 = drolw3 - learning_rate * bdrolw3;
backdrolw4 = drolw4 - learning_rate * bdrolw4;
backdrolb1 = drolb1 - learning_rate * bdrolb1;
backdrolb2 = drolb2 - learning_rate * bdrolb2;
backdrolb3 = drolb3 - learning_rate * bdrolb3;
backdrolb4 = drolb4 - learning_rate * bdrolb4;
    end
end
end
end
end


%% plotting
M_normalised = zeros(floor(Nt / recordInterval), Nz);
for i = 1 : floor(Nt / recordInterval)
for j = 1 : Nz
M_normalised(i, j) = (Ce_new(i,j)*(1-Xc_new(i,j))+omega*Xc_new(i,j))*M0/(Nchain0+Rs_new(i,j)-Col_new(i,j)/m)/Mn0;
S(i,j) = (Rol_new(i,j)-Col_new(i,j))/Ce0;
end
end

%%
figure(1)
MW = real(dz/L*trapz(M_normalised,2));
plot(t, MW*100);
hold on
time = [60,49.97,40.01,34.95,29.97,24.9,19.92,14.94,9.88,4.9,0];
point = [37.73,45.69,55.56,62.79,67.34,75.14,83.31,91.88,96.05,98.31,100];
plot(time,point,'ro')
hold on
ML = real(dz/L*trapz(S,2)*100);
plot(t, ML);
hold on
plot(time2,point2,'rsquare')
hold on
time3 = [0,19.94,39.93,60];
point3 = [56.5,58.89,63.39,65.4];
plot(time3,point3,'rdiamond')
hold on
Xc = real(dz/L*trapz(Xc_new,2)*100);
plot(t,Xc)
xlabel('time(days)')
ylim([0 100])


%%
function [droldta4_all, droldta3_all, droldta2_all, droldta1_all] = calculatedrol(drolw1, drolb1, drolw2, drolb2, drolw3, drolb3, drolw4, drolb4, Rs_new, Ce_new, dRsdt_diff_mat, Nz, Nt, recordInterval)
droldta4_all = zeros(floor(Nt / recordInterval), Nz);
droldta3_all = cell(floor(Nt / recordInterval), Nz);
droldta2_all = cell(floor(Nt / recordInterval), Nz); % 使用 cell 数组
droldta1_all = cell(floor(Nt / recordInterval), Nz); % 使用 cell 数组
for i = 1:floor(Nt / recordInterval)
for j = 1:Nz
dRoldtann = [Rs_new(i,j)/36.483,Ce_new(i,j)/17300,dRsdt_diff_mat(i,j)/203.486];
droldtz1 =  dRoldtann * drolw1 + drolb1';
droldta1 = max(0.01*droldtz1, droldtz1);
droldtz2 = droldta1 * drolw2 + drolb2';
droldta2 = 1./(1+exp(-droldtz2));
droldtz3 = droldta2 * drolw3 + drolb3';
droldta3 = max(0.01*droldtz3,droldtz3);
droldtz4 = droldta3*drolw4 + drolb4;
dRoldt = real(droldtz4*22.2);

% 存储值
droldta4_all(i, j) = dRoldt;
droldta3_all{i, j} = droldta3;
droldta2_all{i, j} = droldta2; % 使用花括号进行赋值
droldta1_all{i, j} = droldta1; % 使用花括号进行赋值
end
end
end
%%
function d = leaky_relu_derivative(x, alpha)
d = ones(size(x));
d(x <= 0) = alpha;
end