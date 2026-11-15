%RUN_E2C_NUISANCE  E2c v2: HK P1 with a DEFLATED non-gas nuisance channel.
%
%   v1 result (Steve, 2026-11-02): the free nuisance removes the infeasible
%   part (g 3 -> 0.03: misfit 0.258 -> 0.138 = null level, bound 0.036 ->
%   0.001) and keeps the injected A<->P transfer (inc 6.42/6.30 -> 6.02/5.96),
%   BUT it also absorbs in-phase gas: TV HK/GN 0.79 -> 0.57 -> 0.33. The
%   movement / non-lung columns are partly collinear with the data pattern of
%   in-phase ventilation (electrodes move outward with inspiration).
%   v2: the nuisance is DEFLATED - projected orthogonal to the data-space
%   images of the low-order in-phase gas maps Ac*[w, w.*y_AP, w.*x_LR, w.*1_left]
%   - so it cannot represent ventilation or its low-order regional
%   heterogeneity. Also v2: the cleaned anti-phase loss is normalized by the
%   UNCLEANED TV (v1 used the cleaned TV, which the nuisance itself shrank),
%   and the A>P / P>A floors are measured with in-phase nulls of BOTH gradient
%   signs (null-g+: anterior +30 %, null-g-: posterior +30 %).
%   Grid g in {3, 0.3} x {real, inj 10, null-g+, null-g-}: 8 runs, ~1 h.
%
%   ---- v1 description ----
%
%   diag_E2_antiphase: the real neonate data contain a component that is
%   anti-correlated with the global signal at ZERO lag (corr -0.94), worth
%   51 %TV of "gas loss" during inspiration (in-phase null: 6.9 %TV), located
%   in non-lung tissue near the chest wall (28 %TV, 21 of it peripheral) and
%   in the central lung (21 %TV); its centroid is at the thorax centre, NOT
%   at the heart. Zero lag + mirror image of inspiration = signature of
%   boundary / electrode movement or of non-lung tissue changes, not of
%   pendelluft. The lung-only HK graph must squeeze it into lung nodes ->
%   positivity bound active (5 %) and misfit 0.318 (in-phase null: 0.138).
%
%   Nuisance (no positivity, no transport, Tikhonov gamma):
%     Y ~ (-J) EL rho + eta_op eta,  eta_op = [J Phi_nl, J_mov]
%       Phi_nl: K1 voxel clusters of the NON-LUNG tissue
%       J_mov : in-plane (x, y) electrode-movement Jacobian, 16 x 2 columns
%               (finite differences as EIDORS jacobian_movement_perturb)
%     every column scaled to the median column norm of the lung operator, so
%     gamma = g * c0n^2 is relative (g >> 1: nuisance off, g << 1: free).
%   Grid g in {3, 0.3, 0.03}; for each g: real data and real + inj 10 %TV
%   (same injected transient transfer as run_E2b_floor).
%   Guards (the nuisance must not eat the gas signal):
%     TV ratio HK/GN ~ 1 and the inj-10 increment ~ 6.4 %TV (no-nuisance
%     value in E2b: P>A 6.46, A>P 6.30).
%   Reported per run: misfit, positivity bound fraction, TV ratio, HK P>A and
%   A>P (%TV), share of ||Y|| explained by each nuisance block, and the
%   anti-phase loss of the element GN image of the CLEANED data
%   Y - eta_op*eta (real 51.0 %TV before cleaning, in-phase null 6.9 %TV).
%   Baseline without nuisance (E2b): misfit 0.318, bound 0.052, P>A 0.70,
%   A>P 2.03 %TV.
%
%   ~1 h (6 HK runs with the dense nuisance solve). Needs hk_core v7.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
fn = fullfile(here, 'Data', 'baby', 'if-neonate-spontaneous (1)', 'P04P-1016.get');

P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 6000;
K1 = 40;  gs = [3 0.3];  dose = 10;  cons_min = 0.75;  deflate = true;  grad_amp = 0.3;

% ---------------- data, filter, model (as run_E2b_floor) ----------------
D   = load_real_eit(fn);
fps = D.fps;  dt = 1/fps;  V = D.vv;  T = size(V, 2);
g   = sum(V ./ mean(V, 2) - 1, 1);
[S_, fr] = spectrum(g, fps);
RR  = peak_in(S_, fr, [0.4 1.6]);  HR = peak_in(S_, fr, [1.7 3.5]);
fc  = min(0.8*HR, max(2.2*RR, 0.5*(2*RR + 0.8*HR)));
V_lp = fft_lowpass(V, fps, fc, 0.2*fc);

fm = mk_library_model('neonate_16el_lungs');
fm.electrode = fm.electrode(electrode_order(fm, 'left'));
fm.stimulation = D.stim;  fm = mdl_normalize(fm, 1);
Ne = size(fm.elems, 1);
lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
nl_idx   = setdiff((1:Ne)', lung_idx);
s0 = ones(Ne,1);  s0(lung_idx) = sig_lung;
J  = calc_jacobian(mk_image(fm, s0));
grp = zeros(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
Gc = calc_dual_graph(fm, lung_idx, struct('target_N', N_coarse, 'groups', grp));
Ac = J * Gc.EL;

minsep = round(0.6 * fps / RR);
vh = mean(V_lp, 2);
for pass = 1:2
    Y   = calc_difference_data(vh, V_lp, fm);
    hp  = lcurve_hp(Ac, Y);
    rgn = -gn_frames(Ac, Y, hp);
    gV  = Gc.vol' * rgn;
    kmin = find(islocalmin(gV, 'MinSeparation', minsep, 'MinProminence', 0.2*p2p(gV)));
    vh  = mean(V_lp(:, kmin), 2);
end
nb = numel(kmin) - 1;
kei = zeros(nb, 1);  tivm = zeros(Gc.N, 1);  TVg = zeros(nb, 1);
for b = 1:nb
    kk = kmin(b):kmin(b+1);  [~, ip] = max(gV(kk));  kei(b) = kk(ip);
    tivm = tivm + (rgn(:, kei(b)) - rgn(:, kk(1))) / nb;
    TVg(b) = gV(kei(b)) - gV(kk(1));
end
[ant_dir, rgt_dir, c_t] = axes_dir(fm);
ant_c = (Gc.cent(:,1:2) - c_t) * ant_dir' > 0;
VolA  = sum(Gc.vol(ant_c));  VolP = sum(Gc.vol(~ant_c));
L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/VolA - sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/VolP);
w_tidal = max(tivm, 0);
rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);

% injected transfer (as run_E2b_floor)
pat  = double(ant_c) / VolA - double(~ant_c) / VolP;
drho = zeros(Gc.N, T);
for b = 1:nb
    ki = kmin(b):kei(b);  n = numel(ki);
    drho(:, ki) = (dose/100 * TVg(b)) * pat * sin(pi * (0:n-1) / (n-1));
end
Yinj = Y - Ac*drho;

% in-phase nulls with an A-P amplitude gradient of both signs (as run_E2b_floor)
E   = Y + Ac * rgn;
yap = (Gc.cent(:,1:2) - c_t) * ant_dir';   yg = yap / max(abs(yap));
xlr = (Gc.cent(:,1:2) - c_t) * rgt_dir';   xg = xlr / max(abs(xlr));
wgp = w_tidal .* (1 + grad_amp*yg);   wgm = w_tidal .* (1 - grad_amp*yg);
Ygp = -Ac*(wgp * (gV / (Gc.vol'*wgp))) + phase_rand(E, 3);
Ygm = -Ac*(wgm * (gV / (Gc.vol'*wgm))) + phase_rand(E, 5);

% ---------------- nuisance operator ----------------
t0 = tic;
Gn  = calc_dual_graph(fm, nl_idx, struct('target_N', K1));
Jnl = J * Gn.EL;                                 % M x K1 (non-lung tissue clusters)
Jm  = movement_jac(fm, s0, [1 2], 1e-6);         % M x 32 (in-plane electrode movement)
c0  = median(sqrt(sum(Ac.^2, 1)));
cn  = @(X) X ./ max(sqrt(sum(X.^2, 1)), eps) * c0;
Bnl = cn(Jnl);  Bmv = cn(Jm);
eta_op = [Bnl, Bmv];
if deflate
    % protected subspace: data images of the low-order in-phase gas maps
    isLc = (Gc.EL' * double(grp == 1)) ./ max(full(sum(Gc.EL, 1))', 1) > 0.5;
    Wv   = [w_tidal, w_tidal.*yg, w_tidal.*xg, w_tidal.*isLc];
    [Uv, Sv, ~] = svd(Ac * Wv, 'econ');  Uv = Uv(:, diag(Sv) > 1e-8*Sv(1));
    e0 = norm(eta_op, 'fro');
    eta_op = eta_op - Uv * (Uv' * eta_op);
    Bnl = eta_op(:, 1:size(Bnl,2));  Bmv = eta_op(:, size(Bnl,2)+1:end);
    fprintf('deflation: %d protected directions, %.1f %% of the nuisance energy removed\n', ...
            size(Uv,2), 100*(1 - norm(eta_op,'fro')^2/e0^2));
end
sJ  = normest(J, 1e-4);  c0n = c0 / sJ;
fprintf('nuisance: %d non-lung clusters + %d movement columns (%.0f s) | rank %d of M = %d | c0n %.3g\n', ...
        Gn.N, size(Bmv,2), toc(t0), rank(eta_op), size(J,1), c0n);

% ---------------- element GN anti-phase metric (as diag_E2_antiphase) ----------------
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
zel = mean(cell2mat(arrayfun(@(e) fm.nodes(e.nodes,3), fm.electrode(:), 'UniformOutput', false)));
H   = max(fm.nodes(:,3)) - min(fm.nodes(:,3));
slab = abs(ce(:,3) - zel) < 0.25 * H;
hpe  = lcurve_hp(J, Y);
apl  = @(Yx) antiphase_loss(-gn_frames(J, Yx, hpe), kmin, kei, slab, vol, cons_min);
[L0, ~, TV0] = apl(Y);
fprintf('anti-phase loss of the element GN image, uncleaned real data: %.1f %%TV (diag_E2_antiphase: 51.0)\n', L0);

% ---------------- runs ----------------
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'dt', dt, ...
           'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', true, 'eta_op', eta_op, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
Kn = size(Bnl, 2);
Rs = struct('g', {}, 'cond', {}, 'Bp', {});
fprintf('\n');
for gg = gs
    o.gamma = gg * c0n^2;
    for c = 1:4
        switch c
            case 1; Yk = Y;    nm = 'real';
            case 2; Yk = Yinj; nm = sprintf('inj %g', dose);
            case 3; Yk = Ygp;  nm = 'null-g+';
            case 4; Yk = Ygm;  nm = 'null-g-';
        end
        t0 = tic;
        [rho, m, eta, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
        Bp  = 100 * breath_metrics(m, Gc, ant_c, kmin, kei, dt) ./ TVg;
        hV  = Gc.vol' * rho;
        TVh = arrayfun(@(b) hV(kei(b)) - hV(kmin(b)), 1:nb)';
        pnl = Bnl * eta(1:Kn, :);  pmv = Bmv * eta(Kn+1:end, :);
        Yc  = Yk - pnl - pmv;
        mis = norm(-Ac*rho + pnl + pmv - Yk, 'fro') / norm(Yk, 'fro');
        act = (rho + rho0) <= 1e-3 * rho0;
        [~, Lc] = apl(Yc);
        fprintf(['g %-5g %-7s | misfit %.3f | bound %.3f | TV HK/GN %.2f | HK P>A %.2f A>P %.2f %%TV | ' ...
                 '||nuis||/||Y||: non-lung %.2f movement %.2f | anti-phase loss cleaned %.1f %%TV(uncleaned) | res %.0e %d it %.0f s\n'], ...
                 gg, nm, mis, mean(act(:)), median(TVh ./ TVg), median(Bp(:,1), 'omitnan'), ...
                 median(Bp(:,2), 'omitnan'), norm(pnl,'fro')/norm(Yk,'fro'), norm(pmv,'fro')/norm(Yk,'fro'), ...
                 100*Lc/TV0, max(info.res), info.iters, toc(t0));
        Rs(end+1) = struct('g', gg, 'cond', nm, 'Bp', Bp); %#ok<SAGROW>
    end
end
save(fullfile(here, 'E2c_v2_results.mat'), 'Rs', 'gs', 'K1', 'dose', 'TVg', 'kmin', 'kei', 'c0n', 'deflate', 'grad_amp');

fprintf('\n============ E2c: injection sensitivity with the nuisance (expected ~%g x 0.75) ============\n', dose);
fprintf('%-6s %12s %12s   (no nuisance, E2b: 6.46 / 6.30)\n', 'g', 'inc P>A', 'inc A>P');
for gg = gs
    a = Rs([Rs.g] == gg & strcmp({Rs.cond}, 'real'));  b = Rs([Rs.g] == gg & strncmp({Rs.cond}, 'inj', 3));
    fprintf('%-6g %12.2f %12.2f\n', gg, median(b.Bp(:,1) - a.Bp(:,1), 'omitnan'), median(b.Bp(:,2) - a.Bp(:,2), 'omitnan'));
end
fprintf('\n============ E2c: real vs in-phase floor (max of null-g+ / null-g-), %%TV ============\n');
fprintf('%-6s %10s %10s %10s %10s\n', 'g', 'real P>A', 'floor P>A', 'real A>P', 'floor A>P');
for gg = gs
    a  = Rs([Rs.g] == gg & strcmp({Rs.cond}, 'real'));
    nn = Rs([Rs.g] == gg & strncmp({Rs.cond}, 'null', 4));
    fl = max(cell2mat(arrayfun(@(r) median(r.Bp, 1, 'omitnan'), nn(:), 'UniformOutput', false)), [], 1);
    fprintf('%-6g %10.2f %10.2f %10.2f %10.2f\n', gg, median(a.Bp(:,1),'omitnan'), fl(1), median(a.Bp(:,2),'omitnan'), fl(2));
end

% ======================================================================
function [L, Labs, TV] = antiphase_loss(x, kmin, kei, slab, vol, cons_min)
nb = numel(kei);
dI = x(:, kei) - x(:, kmin(1:nb));
TIVe = mean(dI, 2);  cons = mean(dI < 0, 2);
TV  = sum(vol(slab) .* TIVe(slab));
neg = slab & TIVe < 0 & cons >= cons_min;
Labs = sum(-vol(neg) .* TIVe(neg));
L   = 100 * Labs / TV;
end

function Es = phase_rand(E, seed)
rng(seed);
T = size(E, 2);  nh = floor((T-1)/2);
ph = exp(1i * 2*pi * rand(1, nh));
P  = ones(1, T);  P(2:nh+1) = ph;  P(T-nh+1:T) = conj(fliplr(ph));
Es = real(ifft(fft(E, [], 2) .* P, [], 2));
end

function Jm = movement_jac(fm, s0, dims, delta)
% finite-difference electrode-movement Jacobian, normalized like J (dV/V)
img  = mk_image(fm, s0);
d0   = fwd_solve(img);  v0 = d0.meas;
node0 = fm.nodes;  ne = numel(fm.electrode);
Jm = zeros(numel(v0), ne * numel(dims));  col = 0;
for d = dims
    for i = 1:ne
        idx = fm.electrode(i).nodes;
        img.fwd_model.nodes = node0;
        img.fwd_model.nodes(idx, d) = node0(idx, d) + delta;
        di = fwd_solve(img);
        col = col + 1;
        Jm(:, col) = (di.meas - v0) / delta ./ v0;
    end
end
end

function B = breath_metrics(m, Gc, sel, kmin, kei, dt)
nb = numel(kei);  B = nan(nb, 2);
QG = hk_pendelluft(Gc, m, sel, dt);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    if numel(st) < 2; continue; end
    B(b,:) = [dt*sum(max(QG(st), 0)), dt*sum(max(-QG(st), 0))];
end
end

function v = p2p(x); v = max(x) - min(x); end

function [S, f] = spectrum(g, fps)
g = g(:)';  T = numel(g);
g = g - polyval(polyfit(1:T, g, 1), 1:T);
w = 0.5 - 0.5*cos(2*pi*(0:T-1)/(T-1));
nf = 8*2^nextpow2(T);
S = abs(fft(g .* w, nf));  f = (0:nf-1) / nf * fps;
keep = f <= fps/2;  S = S(keep);  f = f(keep);
end

function fpk = peak_in(S, f, band)
in = f >= band(1) & f <= band(2);  fi = f(in);  [~, k] = max(S(in));  fpk = fi(k);
end

function Y = fft_lowpass(X, fps, fc, tw)
T  = size(X, 2);
s  = (0:T-1) / (T-1);
tr = X(:,1) + (X(:,end) - X(:,1)) .* s;
f  = abs([0:floor(T/2), -ceil(T/2)+1:-1]) * fps / T;
H  = double(f <= fc - tw/2);
m  = f > fc - tw/2 & f < fc + tw/2;
H(m) = 0.5 * (1 + cos(pi * (f(m) - (fc - tw/2)) / tw));
Y  = real(ifft(fft(X - tr, [], 2) .* H, [], 2)) + tr;
end

function [ant, rgt, c_t] = axes_dir(fm)
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
c_t = wc(1:size(fm.elems,1));  c_l = wc(fm.mat_idx{2});  c_r = wc(fm.mat_idx{3});
rgt = (c_r - c_l) / norm(c_r - c_l);
ant = c_t - (c_l + c_r)/2;  ant = ant - dot(ant, rgt)*rgt;  ant = ant / norm(ant);
end

function order = electrode_order(fm, towards)
[ant, rgt, c_t] = axes_dir(fm);
ep  = zeros(numel(fm.electrode), 2);
for k = 1:numel(fm.electrode); ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes, 1:2), 1) - c_t; end
a   = atan2(ep*rgt', ep*ant');
[~, k0] = min(abs(a));  n = size(ep, 1);
order = circshift(1:n, -(k0-1));
side = dot(ep(order(5),:), rgt);
if (strcmp(towards, 'left') && side > 0) || (strcmp(towards, 'right') && side <= 0)
    order = order([1, n:-1:2]);
end
end

function hp = lcurve_hp(A, Y)
p = 1 ./ sqrt(sum(A.^2, 1))';
S = A * (p .* A');  S = (S + S')/2;
[U, D] = eig(S);  s = max(diag(D), 0);
Z2 = sum((U' * Y).^2, 2);
lams = logspace(-8, 2, 101) * mean(s);
rr = zeros(size(lams));  xx = zeros(size(lams));
for i = 1:numel(lams)
    rr(i) = sum((lams(i) ./ (s + lams(i))).^2 .* Z2);
    xx(i) = sum(s .* Z2 ./ (s + lams(i)).^2);
end
x = 0.5*log(rr);  y = 0.5*log(xx);  t = log(lams);
x1 = gradient(x, t);  y1 = gradient(y, t);  x2 = gradient(x1, t);  y2 = gradient(y1, t);
kap = (x1.*y2 - x2.*y1) ./ max((x1.^2 + y1.^2).^1.5, eps);
kap([1:3, end-2:end]) = -Inf;
[~, k] = max(kap);  hp = sqrt(lams(k));
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end
