
clear all; close all; clc;

%% ============================================================
%  仿真参数
%% ============================================================
dt         = 0.001;
simul_time = 30.0;
n_steps    = round(simul_time / dt);

anim_every = 20;      % 动画降采样: 20 -> 50 FPS

%% ============================================================
%  1. Husky 几何 / 运动学参数
%% ============================================================
r = 0.1651;
a = 0.2854;
l = 0.256;

Omega = [1/r,  a/r;
         1/r, -a/r];

% 动画用车身尺寸
body_L = 0.99;
body_W = 0.67;
wheel_r_vis = 0.1651;

%% ============================================================
%  2. 参考轨迹
%% ============================================================
R_path = 4.0;
wr     = 0.3;

%% ============================================================
%  3. 控制器参数（整定后）
%% ============================================================
% 位置环
P.K_R1 = 5.0;   P.K_R2 = 5.0;   P.P1 = 0.5;  P.P2 = 2.0;

% 姿态环 (理论版: 前馈 + ρ加权双幂次, 无死区分段)
P.K_A0 = 20.0;
P.K_A1 = 30.0;  P.K_A2 = 10.0;

% ---- 指令限幅 ----
P.V_MAX = 5.0;    % [m/s]
P.W_MAX = 10.0;   % [rad/s]

% FTDSO
P.K_o1 = 30.0;  P.K_o2 = 20.0;
P.p_o1 = 1.0;   P.p_o2 = 1.2;

% 内环轮速 PID (含积分)
P.kp_w = 50.0;  P.ki_w = 10.0;  P.kd_w = 0.5;
P.int_w_max    = 4.0;
P.d_filt_alpha = 0.95;
P.tau_max      = 100.0;

% 死区补偿力矩 (符号: 右+ 左-)
P.tau_DZ = 20.0;

%% ============================================================
%  4. 简化被控对象参数 (车体级动力学)
%% ============================================================
PLANT.m_t     = 46.034 + 0.1 + 0.1 + 4*2.637;               % 总质量
PLANT.I_zz    = 2.0296 + 4*(0.02467 + 2.637*(a^2 + l^2));   % 总偏航惯量
PLANT.c_v     = 100.0;    % 平移粘性阻尼
PLANT.c_omega = 50.0;     % 偏航粘性阻尼

%% ============================================================
%  5. 状态初始化 (原点出发, 静止)
%% ============================================================
x = 0; y = 0; theta = 0;
v = 0; omega = 0;

obs_state = [R_path; 0];        % FTDSO 初值 [R_path, 0]
dPhi_filt = zeros(2,1);
Phi_prev  = zeros(2,1);
int_w     = zeros(2,1);         % 内环积分状态

%% ============================================================
%  6. 日志容器
%% ============================================================
q_log     = zeros(3, n_steps);
qr_log    = zeros(3, n_steps);
qe_log    = zeros(3, n_steps);
tau_log   = zeros(2, n_steps);   % PID 输出 (补偿前)
tauc_log  = zeros(2, n_steps);   % 补偿后实际指令
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
intw_log  = zeros(2, n_steps);   % 内环积分状态

%% ============================================================
%  7. 动画窗口初始化
%% ============================================================
fig_anim = figure('Name','Husky 实时动画 (DS_FD)','Color','w', ...
                  'Position',[100 100 900 750]);

ax = axes('Parent', fig_anim);
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
xlabel(ax, 'X [m]'); ylabel(ax, 'Y [m]');
title(ax, 'Husky (FTDSO + \rho\alpha_{til} + 内环积分 + 死区补偿)');

xlim(ax, [-R_path-1.5, R_path+1.5]);
ylim(ax, [-R_path-1.5, R_path+1.5]);

% 参考圆
th_vis = linspace(0, 2*pi, 200);
plot(ax, R_path*cos(th_vis), R_path*sin(th_vis), ...
     'r-', 'LineWidth', 1.5);

% 参考点（黄球）
h_ref_pt = plot(ax, R_path, 0, 'o', ...
    'MarkerSize', 12, 'MarkerFaceColor', [1 1 0], ...
    'MarkerEdgeColor', 'k');

% 实际轨迹
h_traj = plot(ax, x, y, 'b--', 'LineWidth', 1.5);

% 误差连线
h_err = plot(ax, [R_path, x], [0, y], '-', ...
    'Color', [0.5 0.5 0.5], 'LineWidth', 1.0);

% 车体 patch
body_pts = [-body_L/2, -body_W/2;
             body_L/2, -body_W/2;
             body_L/2,  body_W/2;
            -body_L/2,  body_W/2]';
h_body = patch(ax, 'XData', body_pts(1,:), 'YData', body_pts(2,:), ...
    'FaceColor', [0.2 0.4 0.8], 'FaceAlpha', 0.7, ...
    'EdgeColor', 'k', 'LineWidth', 1.5);

% 车头方向线
head_len = body_L/2;
h_head = plot(ax, [0, head_len], [0, 0], 'k-', 'LineWidth', 2);

% 四个轮子
wheel_pos_local = [ l/2, -a;      % 前左
                    l/2,  a;      % 前右
                   -l/2, -a;      % 后左
                   -l/2,  a];     % 后右
h_wheels = gobjects(4,1);
for iw = 1:4
    h_wheels(iw) = rectangle(ax, 'Position', ...
        [-wheel_r_vis, -wheel_r_vis, 2*wheel_r_vis, 2*wheel_r_vis], ...
        'Curvature', [1 1], 'FaceColor', [0.3 0.3 0.3], ...
        'EdgeColor', 'k');
end

% 状态文字
h_txt = text(ax, 0, 0, '', 'FontSize', 11, ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

legend(ax, [h_ref_pt, h_traj, h_err], ...
    {'参考点','机器人轨迹','误差连线'}, 'Location','northwest');

drawnow;

%% ============================================================
%  8. 主循环
%% ============================================================
for k = 1:n_steps
    t = (k-1) * dt;

    % ---- 由车体速度逆解左右轮角速度 ----
    phi_R = v / r + a * omega / r;
    phi_L = v / r - a * omega / r;
    Phi   = [phi_R; phi_L];

    q = [x; y; theta];

    % ---- 参考轨迹 ----
    qr = [R_path*cos(wr*t);
          R_path*sin(wr*t);
          wr*t + pi/2];

    % ---- 外环: FTDSO + 理论版姿态环 ----
    [Vref, qe, obs_state, dbg] = cascade_ftc(qr, q, obs_state, dt, P);

    % ---- 逆运动学 ----
    Phi_ref = Omega * Vref;

    % ---- 内环 PID (含积分) ----
    [tau, dPhi_filt, int_w] = wheel_speed_pid(Phi_ref, Phi, ...
        dPhi_filt, Phi_prev, int_w, dt, P);
    Phi_prev = Phi;

    % ---- 死区补偿力矩 (符号: 右+ 左-) ----
    w_c = Vref(2);
    if     w_c >  1e-3, sgn =  1.0;
    elseif w_c < -1e-3, sgn = -1.0;
    else,               sgn =  0.0;
    end
    tau_comp  = P.tau_DZ * sgn;
    tau_R_cmd = max(min(tau(1) + 0.5*tau_comp,  P.tau_max), -P.tau_max);
    tau_L_cmd = max(min(tau(2) - 0.5*tau_comp,  P.tau_max), -P.tau_max);
    tau_cmd   = [tau_R_cmd; tau_L_cmd];

    % ---- 车体级简化动力学 ----
    F_total = (tau_cmd(1) + tau_cmd(2)) / r;      % 总纵向推力
    M_total = a * (tau_cmd(1) - tau_cmd(2)) / r;  % 总偏航力矩

    dv     = (F_total - PLANT.c_v     * v)     / PLANT.m_t;
    domega = (M_total - PLANT.c_omega * omega) / PLANT.I_zz;

    v     = v     + dv     * dt;
    omega = omega + domega * dt;

    x     = x + v * cos(theta) * dt;
    y     = y + v * sin(theta) * dt;
    theta = wrap_to_pi(theta + omega * dt);

    % ---- 记录 ----
    q_log(:, k)    = [x; y; theta];
    qr_log(:, k)   = qr;
    qe_log(:, k)   = qe;
    tau_log(:, k)  = tau;
    tauc_log(:, k) = tau_cmd;
    vref_log(:, k) = Vref;
    vest_log(:, k) = [dbg.vrx_est; dbg.vry_est];
    thff_log(k)    = dbg.theta_ff;
    thfb_log(k)    = dbg.theta_fb;
    alpha_log(k)   = dbg.alpha;
    atil_log(k)    = dbg.alpha_til;
    aff_log(k)     = dbg.alpha_ff;
    b_log(k)       = dbg.b;
    beta_log(k)    = dbg.beta;
    rho_log(k)     = dbg.rho;
    vd_log(k)      = dbg.v_d;
    intw_log(:, k) = int_w;

    % ============================================================
    % 动画刷新
    % ============================================================
    if mod(k, anim_every) == 0
        R_b = [cos(theta), -sin(theta);
               sin(theta),  cos(theta)];
        world_pts = R_b * body_pts + [x; y];
        set(h_body, 'XData', world_pts(1,:), 'YData', world_pts(2,:));

        head_world = R_b * [0, head_len; 0, 0] + [x; y];
        set(h_head, 'XData', head_world(1,:), 'YData', head_world(2,:));

        for iw = 1:4
            pos_world = R_b * wheel_pos_local(iw,:)' + [x; y];
            set(h_wheels(iw), 'Position', ...
                [pos_world(1)-wheel_r_vis, pos_world(2)-wheel_r_vis, ...
                 2*wheel_r_vis, 2*wheel_r_vis]);
        end

        set(h_ref_pt, 'XData', qr(1), 'YData', qr(2));
        set(h_err, 'XData', [qr(1), x], 'YData', [qr(2), y]);
        set(h_traj, 'XData', q_log(1, 1:k), 'YData', q_log(2, 1:k));

        txt = sprintf(['t = %.2f s\n' ...
                       '\\rho = %.4f m\n' ...
                       'v_c = %+.3f m/s\n' ...
                       '\\omega_c = %+.3f rad/s\n' ...
                       'b = %d (%s)\n' ...
                       'int_w = [+%.2f, +%.2f]'], ...
                       t, dbg.rho, Vref(1), Vref(2), ...
                       dbg.b, ternary(dbg.b==0, 'FWD', 'REV'), ...
                       int_w(1), int_w(2));
        set(h_txt, 'String', txt, ...
            'Position', [-R_path-1.4, R_path+1.4]);

        drawnow limitrate;
    end
end

t_axis = (0:n_steps-1) * dt;

%% ============================================================
%  9. 结束后的静态绘图
%% ============================================================
plot_results(t_axis, q_log, qr_log, tau_log, tauc_log, vref_log, ...
    vest_log, thff_log, thfb_log, alpha_log, atil_log, aff_log, ...
    b_log, beta_log, rho_log, vd_log, intw_log, ...
    R_path, wr, P.tau_max, P.int_w_max, P.tau_DZ);


%% ============================================================
%  局部函数
%% ============================================================

function y = sig(x, p)
    y = abs(x).^p .* sign(x);
end

function y = wrap_to_pi(x)
    y = atan2(sin(x), cos(x));
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

% ------------------------------------------------------------
% 控制器: 外环 + FTDSO + 理论版姿态环
% 无死区分段, 无姿态环积分, 反馈项乘 rho, 输出限幅
% ------------------------------------------------------------
function [Vref, qe, obs_state, dbg] = cascade_ftc(qr, q, obs_state, dt, P)

    ex  = qr(1) - q(1);
    ey  = qr(2) - q(2);
    rho = hypot(ex, ey);

    % ---- FTDSO ----
    xr_hat = obs_state(1);
    yr_hat = obs_state(2);
    e_ox = qr(1) - xr_hat;
    e_oy = qr(2) - yr_hat;
    vrx_est = P.K_o1*sig(e_ox, P.p_o1) + P.K_o2*sig(e_ox, P.p_o2);
    vry_est = P.K_o1*sig(e_oy, P.p_o1) + P.K_o2*sig(e_oy, P.p_o2);
    xr_hat = xr_hat + vrx_est * dt;
    yr_hat = yr_hat + vry_est * dt;
    obs_state = [xr_hat; yr_hat];

    % ---- 期望航向 (无奇异保护, 直接 atan2) ----
    theta_fb = atan2(ey, ex);
    theta_ff = atan2(vry_est, vrx_est);

    % ---- 航向误差 ----
    alpha    = wrap_to_pi(theta_fb - q(3));
    alpha_ff = wrap_to_pi(theta_ff - q(3));

    % ---- 模式切换 (理论判据, 无滞回) ----
    beta = wrap_to_pi(theta_fb - theta_ff);
    if abs(beta) > pi/2
        b = 1;
    else
        b = 0;
    end
    alpha_til = wrap_to_pi(alpha - b*pi);

    % ---- 位置环 ----
    v_d = P.K_R1*sig(rho, P.P1) + P.K_R2*sig(rho, P.P2);
    v_c = v_d * cos(alpha);

    % ---- 姿态环 (理论版: 前馈 + ρ加权双幂次, 无死区分段) ----
    omega_c = P.K_A0 * alpha_ff ...
            + P.K_A1 * sig(rho*alpha_til, P.P1) ...
            + P.K_A2 * sig(rho*alpha_til, P.P2);

    % ---- 指令限幅 ----
    v_c     = max(min(v_c,     P.V_MAX), -P.V_MAX);
    omega_c = max(min(omega_c, P.W_MAX), -P.W_MAX);

    Vref = [v_c; omega_c];

    % ---- 车体系误差记录 ----
    eth = wrap_to_pi(qr(3) - q(3));
    ct = cos(q(3)); st = sin(q(3));
    R = [ ct, st, 0;
         -st, ct, 0;
          0,  0,  1];
    qe = R * [ex; ey; eth];

    % ---- 调试信息 ----
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

% ------------------------------------------------------------
% 内环轮速 PID (含积分)
% ------------------------------------------------------------
function [tau, dPhi_filt, int_w] = ...
        wheel_speed_pid(Phi_ref, Phi, dPhi_filt, Phi_prev, int_w, dt, P)

    err = Phi_ref - Phi;

    % 积分项 (限幅)
    int_w = int_w + err * dt;
    int_w = max(min(int_w, P.int_w_max), -P.int_w_max);

    % 微分项 (一阶滤波)
    dPhi_raw  = (Phi - Phi_prev) / dt;
    dPhi_filt = P.d_filt_alpha * dPhi_filt + ...
                (1 - P.d_filt_alpha) * dPhi_raw;

    tau = P.kp_w*err + P.ki_w*int_w - P.kd_w*dPhi_filt;
    tau = max(min(tau, P.tau_max), -P.tau_max);
end

% ------------------------------------------------------------
% 静态绘图
% ------------------------------------------------------------
function plot_results(t_axis, q_log, qr_log, tau_log, tauc_log, vref_log, ...
    vest_log, thff_log, thfb_log, alpha_log, atil_log, aff_log, ...
    b_log, beta_log, rho_log, vd_log, intw_log, ...
    R_path, wr, tau_max, int_w_max, tau_DZ)

% ---- 轨迹 ----
figure('Name','轨迹跟踪','Position',[100 100 1200 800]);
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
sgtitle('DS\_FD 对应: FTDSO + \rho\alpha_{til} + 内环积分 + 死区补偿 + 限幅 (整定参数)');

% ---- 航向 / 模式 ----
figure('Name','航向与模式','Position',[150 150 1000 800]);
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
ylim([-0.2 1.2]); yticks([0 1]); yticklabels({'前向 FWD','反向 REV'});
xlabel('Time [s]'); title('模式变量 b'); grid on;

% ---- v_d 与 FTDSO ----
figure('Name','v_d 与 FTDSO','Position',[200 200 1000 400]);
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
figure('Name','轮力矩','Position',[250 250 800 500]);
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
title(sprintf('实际指令 (含 \\pm\\tau_{DZ}/2 = \\pm%.0f 死区补偿)', tau_DZ/2));
legend('右轮','左轮'); grid on;

% ---- 内环积分状态 ----
figure('Name','内环积分','Position',[300 300 800 300]);
plot(t_axis, intw_log(1,:), 'b'); hold on;
plot(t_axis, intw_log(2,:), 'r');
yline( int_w_max, 'k--'); yline(-int_w_max, 'k--');
xlabel('Time [s]'); ylabel('\int e_\phi dt [rad]');
title('内环轮速积分状态'); legend('右轮','左轮'); grid on;
end
