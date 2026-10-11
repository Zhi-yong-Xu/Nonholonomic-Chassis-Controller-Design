%% ============================================================
%  DS_FD 严格对应 MATLAB 版本
%  FTDSO + 理论版姿态环(ρ·α_til) + 内环积分 + 死区补偿 + 指令限幅
%  被控对象: 简化轮动力学 + 差速运动学 (替代 MuJoCo husky.xml)
%% ============================================================
clear; clc; close all;

sig        = @(x,p) abs(x).^p .* sign(x);
wrap_to_pi = @(x) atan2(sin(x), cos(x));

%% ============================================================
%  1. Husky 物理参数 (与 DS_FD 一致)
%% ============================================================
r_w = 0.1651;
a_w = 0.2854;
l_w = 0.256;

Omega = [1/r_w,  a_w/r_w;
         1/r_w, -a_w/r_w];

% ---- 简化被控对象 (替代 MuJoCo) ----
% DS.py 中: Iw=0.04411, 粘性摩擦此处取 b_wheel=1.0
% 注: 若需更接近 MuJoCo, 可把车体惯量折算进 I_wheel
I_wheel = 0.05;
b_wheel = 1.0;

%% ============================================================
%  2. 参考轨迹 (DS_FD: R=4.0, wr=0.3)
%% ============================================================
R_path = 4.0;
wr     = 0.3;
vr     = R_path * wr;

%% ============================================================
%  3. 控制器参数 (严格取自 DS_FD)
%% ============================================================
% 位置环
K_R1 = 5.0;  K_R2 = 5.0;  P1 = 0.5;  P2 = 2.0;

% 姿态环（前馈 + ρ加权双幂次反馈, 无死区分段）
K_A0 = 30.0;
K_A1 = 20.0;  K_A2 = 10.0;

% ---- 指令限幅 (DS_FD 特有) ----
V_MAX = 5.0;    % [m/s]
W_MAX = 10.0;   % [rad/s]

% FTDSO
K_o1 = 30.0;  K_o2 = 20.0;
p_o1 = 1.0;   p_o2 = 1.2;

% 内环轮速 PID（含积分, DS_FD 特有）
kp_w = 50.0;  ki_w = 10.0;  kd_w = 0.5;
int_w_max = 4.0;
d_filt_alpha = 0.95;
tau_max = 100.0;

% 死区补偿力矩 (DS_FD 特有, 符号: 右+ 左-)
tau_DZ = 20.0;

%% ============================================================
%  4. 仿真设置
%% ============================================================
dt         = 0.001;
simul_time = 30.0;
n_steps    = round(simul_time / dt);

%% ============================================================
%  5. 初态设置 (对应 DS_FD: 原点出发, 轮速为零)
%    注: MuJoCo 的 2000 步自由落体稳定在此不适用(无重力模型)
%% ============================================================
x_now   = 0.0;
y_now   = 0.0;
yaw_now = 0.0;
phi_R   = 0.0;
phi_L   = 0.0;

% 内环状态
dPhi_filt = zeros(2,1);
Phi_prev  = zeros(2,1);
int_w     = zeros(2,1);      % DS_FD 特有: 内环积分状态

% FTDSO 状态
obs_state = [R_path; 0.0];

%% ============================================================
%  6. 日志容器
%% ============================================================
q_log     = zeros(3, n_steps);
qr_log    = zeros(3, n_steps);
qe_log    = zeros(3, n_steps);
tau_log   = zeros(2, n_steps);    % 记录 PID 输出(补偿前), 与 DS_FD 一致
tauc_log  = zeros(2, n_steps);    % 新增: 补偿后实际指令 (便于检查)
vref_log  = zeros(2, n_steps);
vest_log  = zeros(2, n_steps);
thff_log  = zeros(1, n_steps);
thfb_log  = zeros(1, n_steps);
alpha_log = zeros(1, n_steps);
atil_log  = zeros(1, n_steps);
aff_log   = zeros(1, n_steps);
b_log     = zeros(1, n_steps);
beta_log  = zeros(1, n_steps);
rho_log   = zeros(1, n_steps);
vd_log    = zeros(1, n_steps);
intw_log  = zeros(2, n_steps);    % DS_FD 特有

%% ============================================================
%  7. 主循环
%% ============================================================
for k = 1:n_steps
    t = (k-1) * dt;

    q   = [x_now; y_now; yaw_now];
    Phi = [phi_R; phi_L];

    qr = [R_path*cos(wr*t);
          R_path*sin(wr*t);
          wr*t + pi/2.0];

    % ---- 外环: FTDSO + 理论版姿态环 (与 DS_FD 逐行对应) ----
    [Vref, qe, obs_state, dbg] = cascade_ftc(qr, q, obs_state, dt, ...
        K_o1, K_o2, p_o1, p_o2, ...
        K_R1, K_R2, P1, P2, K_A0, K_A1, K_A2, ...
        V_MAX, W_MAX);

    % ---- 逆运动学 ----
    Phi_ref = Omega * Vref;

    % ---- 内环 PID (含积分, DS_FD 版) ----
    [tau, dPhi_filt, int_w] = wheel_speed_pid(Phi_ref, Phi, ...
        dPhi_filt, Phi_prev, int_w, dt, ...
        kp_w, ki_w, kd_w, int_w_max, d_filt_alpha, tau_max);
    Phi_prev = Phi;

    % ---- 死区补偿力矩 (DS_FD 符号: 右+ 左-) ----
    w_c = Vref(2);
    if     w_c >  1e-3, sgn =  1.0;
    elseif w_c < -1e-3, sgn = -1.0;
    else,               sgn =  0.0;
    end
    tau_comp = tau_DZ * sgn;
    tau_R_cmd = max(min(tau(1) + 0.5*tau_comp,  tau_max), -tau_max);
    tau_L_cmd = max(min(tau(2) - 0.5*tau_comp,  tau_max), -tau_max);

    % ---- 简化轮动力学 (替代 MuJoCo) ----
    tau_cmd = [tau_R_cmd; tau_L_cmd];
    dPhi = (tau_cmd - b_wheel * Phi) / I_wheel;
    phi_R = phi_R + dPhi(1)*dt;
    phi_L = phi_L + dPhi(2)*dt;

    % ---- 差速运动学积分 ----
    v     = r_w/2   * (phi_R + phi_L);
    omega = r_w/(2*a_w) * (phi_R - phi_L);

    x_now   = x_now   + v*cos(yaw_now)*dt;
    y_now   = y_now   + v*sin(yaw_now)*dt;
    yaw_now = wrap_to_pi(yaw_now + omega*dt);

    % ---- 记录 ----
    q_log(:,k)    = q;
    qr_log(:,k)   = qr;
    qe_log(:,k)   = qe;
    tau_log(:,k)  = tau;
    tauc_log(:,k) = tau_cmd;
    vref_log(:,k) = Vref;
    vest_log(:,k) = [dbg.vrx_est; dbg.vry_est];
    thff_log(k)   = dbg.theta_ff;
    thfb_log(k)   = dbg.theta_fb;
    alpha_log(k)  = dbg.alpha;
    atil_log(k)   = dbg.alpha_til;
    aff_log(k)    = dbg.alpha_ff;
    b_log(k)      = dbg.b;
    beta_log(k)   = dbg.beta;
    rho_log(k)    = dbg.rho;
    vd_log(k)     = dbg.v_d;
    intw_log(:,k) = int_w;
end

t_axis = (0:n_steps-1) * dt;

%% ============================================================
%  8. 绘图 (与 DS_FD 图组对应)
%% ============================================================

% ---- 轨迹 ----
figure('Name','轨迹');
subplot(2,2,1);
plot(qr_log(1,:), qr_log(2,:), 'b-', 'LineWidth',1.5); hold on;
plot(q_log(1,:),  q_log(2,:),  'r--','LineWidth',1.5);
xlabel('X [m]'); ylabel('Y [m]');
title('轨迹跟踪'); legend('参考','实际'); grid on; axis equal;

subplot(2,2,2);
plot(t_axis, qr_log(1,:)-q_log(1,:), 'b'); hold on;
plot(t_axis, qr_log(2,:)-q_log(2,:), 'r');
xlabel('Time [s]'); ylabel('误差 [m]');
title('位置误差'); legend('e_x','e_y'); grid on;

subplot(2,2,3);
plot(t_axis, vref_log(1,:), 'b');
xlabel('Time [s]'); ylabel('v [m/s]');
title('线速度指令'); grid on;

subplot(2,2,4);
plot(t_axis, vref_log(2,:), 'b');
xlabel('Time [s]'); ylabel('\omega [rad/s]');
title('角速度指令'); grid on;
sgtitle('DS\_FD 对应: FTDSO + \rho\alpha_{til} + 内环积分 + 死区补偿 + 限幅');

% ---- 航向 / 模式判据 ----
figure('Name','航向与模式');
subplot(2,2,1);
plot(t_axis, thfb_log, 'b'); hold on;
plot(t_axis, thff_log, 'r--');
plot(t_axis, q_log(3,:), 'g:');
xlabel('Time [s]'); ylabel('\theta [rad]');
title('\theta_{fb}, \theta_{ff}, \psi');
legend('\theta_{fb}','\theta_{ff}','\psi'); grid on;

subplot(2,2,2);
plot(t_axis, alpha_log, 'b'); hold on;
plot(t_axis, atil_log,  'r--');
plot(t_axis, aff_log,   'g:');
xlabel('Time [s]'); ylabel('rad');
title('\alpha, \alpha_{til}, \alpha_{ff}');
legend('\alpha','\alpha_{til}','\alpha_{ff}'); grid on;

subplot(2,2,3);
plot(t_axis, beta_log, 'k', 'LineWidth',1.5); hold on;
yline( pi/2, 'r--'); yline(-pi/2, 'r--');
xlabel('Time [s]'); ylabel('\beta [rad]');
title('模式判据 \beta = wrap(\theta_{fb}-\theta_{ff})'); grid on;

subplot(2,2,4);
plot(t_axis, b_log, 'k', 'LineWidth',1.5);
ylim([-0.2 1.2]); yticks([0 1]);
yticklabels({'前向 FWD','反向 REV'});
xlabel('Time [s]'); title('模式变量 b'); grid on;

% ---- v_d 与 FTDSO ----
figure('Name','v_d 与 FTDSO');
subplot(1,2,1);
plot(t_axis, vd_log, 'b', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('v_d [m/s]');
title('位置环输出 v_d'); grid on;

subplot(1,2,2);
plot(t_axis, -R_path*wr*sin(wr*t_axis), 'b'); hold on;
plot(t_axis,  vest_log(1,:),            'r--');
plot(t_axis,  R_path*wr*cos(wr*t_axis), 'g');
plot(t_axis,  vest_log(2,:),            'm--');
xlabel('Time [s]'); ylabel('v [m/s]');
title('FTDSO 速度估计');
legend('v_{rx}真','v_{rx}估','v_{ry}真','v_{ry}估'); grid on;

% ---- 力矩 (PID 输出与补偿后指令) ----
figure('Name','轮力矩');
subplot(2,1,1);
plot(t_axis, tau_log(1,:), 'b'); hold on;
plot(t_axis, tau_log(2,:), 'r');
yline( tau_max, 'k--'); yline(-tau_max, 'k--');
xlabel('Time [s]'); ylabel('\tau [N\cdotm]');
title('PID 输出 (补偿前)'); legend('右轮','左轮'); grid on;

subplot(2,1,2);
plot(t_axis, tauc_log(1,:), 'b'); hold on;
plot(t_axis, tauc_log(2,:), 'r');
yline( tau_max, 'k--'); yline(-tau_max, 'k--');
xlabel('Time [s]'); ylabel('\tau [N\cdotm]');
title('实际指令 (含 \pm\tau_{DZ}/2 死区补偿)'); legend('右轮','左轮'); grid on;

% ---- 内环积分状态 (DS_FD 特有) ----
figure('Name','内环积分');
plot(t_axis, intw_log(1,:), 'b'); hold on;
plot(t_axis, intw_log(2,:), 'r');
yline( int_w_max, 'k--'); yline(-int_w_max, 'k--');
xlabel('Time [s]'); ylabel('\int e_\phi dt [rad]');
title('内环轮速积分状态'); legend('右轮','左轮'); grid on;

%% ============================================================
%  9. 子函数 —— 与 DS_FD 的 cascade_ftc 逐行对应
%% ============================================================
function [Vref, qe, obs_state, dbg] = cascade_ftc(qr, q, obs_state, dt, ...
        K_o1, K_o2, p_o1, p_o2, ...
        K_R1, K_R2, P1, P2, K_A0, K_A1, K_A2, ...
        V_MAX, W_MAX)

    sig        = @(x,p) abs(x).^p .* sign(x);
    wrap_to_pi = @(x) atan2(sin(x), cos(x));

    ex  = qr(1) - q(1);
    ey  = qr(2) - q(2);
    rho = hypot(ex, ey);

    % ---- FTDSO ----
    xr_hat = obs_state(1);
    yr_hat = obs_state(2);
    e_ox = qr(1) - xr_hat;
    e_oy = qr(2) - yr_hat;
    vrx_est = K_o1*sig(e_ox, p_o1) + K_o2*sig(e_ox, p_o2);
    vry_est = K_o1*sig(e_oy, p_o1) + K_o2*sig(e_oy, p_o2);
    xr_hat = xr_hat + vrx_est * dt;
    yr_hat = yr_hat + vry_est * dt;
    obs_state = [xr_hat; yr_hat];

    % ---- 期望航向 (DS_FD: 无奇异保护, 直接 atan2) ----
    theta_fb = atan2(ey, ex);
    theta_ff = atan2(vry_est, vrx_est);

    % ---- 航向误差 ----
    alpha    = wrap_to_pi(theta_fb - q(3));
    alpha_ff = wrap_to_pi(theta_ff - q(3));

    % ---- 模式切换 (理论判据, 无滞回) ----
    beta = wrap_to_pi(theta_fb - theta_ff);
    if abs(beta) > pi/2,  b = 1;  else,  b = 0;  end
    alpha_til = wrap_to_pi(alpha - b*pi);

    % ---- 位置环 ----
    v_d = K_R1*sig(rho, P1) + K_R2*sig(rho, P2);
    v_c = v_d * cos(alpha);

    % ---- 姿态环 (理论版: 前馈 + ρ加权双幂次, 无死区分段) ----
    omega_c = K_A0 * alpha_ff ...
            + K_A1 * sig(rho*alpha_til, P1) ...
            + K_A2 * sig(rho*alpha_til, P2);

    % ---- 指令限幅 (DS_FD 特有) ----
    v_c     = max(min(v_c,     V_MAX), -V_MAX);
    omega_c = max(min(omega_c, W_MAX), -W_MAX);

    Vref = [v_c; omega_c];

    % ---- 误差记录 ----
    eth = wrap_to_pi(qr(3) - q(3));
    ct = cos(q(3));  st = sin(q(3));
    R = [ ct,  st, 0;
         -st,  ct, 0;
          0,   0,  1];
    qe = R * [ex; ey; eth];

    dbg.alpha     = alpha;
    dbg.alpha_til = alpha_til;
    dbg.alpha_ff  = alpha_ff;
    dbg.theta_ff  = theta_ff;
    dbg.theta_fb  = theta_fb;
    dbg.b         = b;
    dbg.beta      = beta;
    dbg.vrx_est   = vrx_est;
    dbg.vry_est   = vry_est;
    dbg.rho       = rho;
    dbg.v_d       = v_d;
end

%% ============================================================
%  10. 子函数 —— 内环 PID (含积分, 与 DS_FD 逐行对应)
%% ============================================================
function [tau, dPhi_filt, int_w] = wheel_speed_pid(Phi_ref, Phi, ...
        dPhi_filt, Phi_prev, int_w, dt, ...
        kp_w, ki_w, kd_w, int_w_max, d_filt_alpha, tau_max)

    err = Phi_ref - Phi;

    % 积分项 (限幅)
    int_w = int_w + err * dt;
    int_w = max(min(int_w, int_w_max), -int_w_max);

    % 微分项 (一阶滤波)
    dPhi_raw  = (Phi - Phi_prev) / dt;
    dPhi_filt = d_filt_alpha*dPhi_filt + (1.0 - d_filt_alpha)*dPhi_raw;

    tau = kp_w*err + ki_w*int_w - kd_w*dPhi_filt;
    tau = max(min(tau, tau_max), -tau_max);
end
