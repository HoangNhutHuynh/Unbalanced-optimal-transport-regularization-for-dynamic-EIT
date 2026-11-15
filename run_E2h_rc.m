%RUN_E2H_RC  E2h: delay or time constant? RC-heterogeneity nulls.
%
%   E2g v2 (pure-delay nulls, zero gas exchange): the timing signature of the
%   quadrature flux identifies the sign of the asynchrony unambiguously:
%     anterior leads (tau < 0): P>A early, A>P late
%     anterior lags  (tau > 0): A>P early, P>A late   <- real data
%   and the size: |tau| 12.8 ms -> 0.67/0.71, 32.7 ms -> 1.56/1.86 %TV.
%   But a pure delay is SYMMETRIC (A>P ~ P>A within an inspiration), the
%   real data are not: A>P 1.73 vs P>A 0.76 (ratio 2.3). A region that fills
%   with a LONGER TIME CONSTANT (first-order RC) lags early and has not fully
%   caught up at end-inspiration, so the return flux falls partly into
%   expiration -> asymmetric within inspiration.
%   Test: RC nulls, zero gas exchange: each region follows the global profile
%   through H(f) = 1/(1 + i 2 pi f tau_RC):
%     anterior slower: tau_A in {15, 30, 60} ms;  posterior slower: tau_P 30 ms
%   Per condition (and for the real data): quadrature PV in inspiration AND in
%   expiration, both directions, timing by thirds of inspiration, asymmetry
%   A>P / P>A in inspiration.
%   (The GN regional-lag estimator is not used: E2g v2 showed it is biased and
%   non-monotonic below one frame, e.g. true -32.7 / -12.8 ms -> -16.6 / -19.6.)
%
%   ~50 min (5 HK runs). Results in E2h_results.mat.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
fn = fullfile(here, 'Data', 'baby', 'if-neonate-spontaneous (1)', 'P04P-1016.get');

P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 6000;  cons_min = 0.75;
MASK = 'thorax';   % 'lungs' reproduces the lung-mask log in README; 'thorax' (after E2i) writes *_thorax.mat

% ---------------- data, filter, model (as run_E2c_nuisance) ----------------
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
s0 = ones(Ne,1);  s0(lung_idx) = sig_lung;
J  = calc_jacobian(mk_image(fm, s0));
grp = 3 * ones(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
[Gc, sfx] = domain_graph(fm, grp, N_coarse, MASK);
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
tl   = max(sum(Gc.vol .* tivm) / sum(Gc.vol), eps);   % mean TIV per unit volume
E    = Y + Ac * rgn;                                   % real GN residual
rho_null = w_tidal * (gV / (Gc.vol' * w_tidal));       % zero pendelluft

% element-GN anti-phase metric (as diag_E2_antiphase), nominal model
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
zel = mean(cell2mat(arrayfun(@(e) fm.nodes(e.nodes,3), fm.electrode(:), 'UniformOutput', false)));
H   = max(fm.nodes(:,3)) - min(fm.nodes(:,3));
slab = abs(ce(:,3) - zel) < 0.25 * H;
hpe  = lcurve_hp(J, Y);
apl  = @(Yx) antiphase_loss(-gn_frames(J, Yx, hpe), kmin, kei, slab, vol, cons_min);
fprintf('real data: element-GN anti-phase loss %.1f %%TV | cluster-GN min gas / mean TIV %.1f\n', ...
        apl(Y), min(rgn(:)) / tl);

% ---------------- conditions ----------------
s_glob = gV / (Gc.vol' * w_tidal);
wA = w_tidal .* ant_c;   wP = w_tidal .* ~ant_c;
C = struct('name', {}, 'Y', {});
C(end+1) = struct('name', 'real', 'Y', Y);
tAs = [0.015 0.030 0.060];
if ~strcmp(MASK, 'lungs'); tAs(end+1) = 0.100; end   % E2i: whole-thorax real PV sits at the lung-mask 60 ms point -> bracket it
for tA = tAs
    rho_rc = wP * s_glob + wA * rc_filter(s_glob, fps, tA);
    C(end+1) = struct('name', sprintf('ant tau %2.0f ms', 1e3*tA), 'Y', -Ac*rho_rc + phase_rand(E, 60 + round(1e3*tA))); %#ok<SAGROW>
end
rho_rc = wP * rc_filter(s_glob, fps, 0.030) + wA * s_glob;
C(end+1) = struct('name', 'post tau 30 ms', 'Y', -Ac*rho_rc + phase_rand(E, 99));

% ---------------- runs ----------------
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
R = nan(numel(C), 11);
fprintf('%-15s | %-23s | %-23s | %-34s | %s\n', 'condition', 'inspiration P>A  A>P', 'expiration P>A  A>P', ...
        'insp thirds P>A (e m l) A>P (e m l)', 'asym A>P/P>A');
for k = 1:numel(C)
    t0 = tic;
    [q, ~, Bi] = run_quad(J, C(k).Y, Gc, o, ant_c, kmin, kei, dt, TVg);
    Be = 100 * exp_pv(q, kmin, kei, dt) ./ TVg;
    TTk = timing_thirds(q, kmin, kei, dt, TVg);
    mi = median(Bi, 1, 'omitnan');  me = median(Be, 1, 'omitnan');
    R(k,:) = [mi, me, TTk, mi(2)/max(mi(1), eps)];
    fprintf('%-15s | %10.2f %10.2f   | %10.2f %10.2f   | %5.2f %5.2f %5.2f  %5.2f %5.2f %5.2f | %5.2f  (%.0f s)\n', ...
            C(k).name, R(k,:), toc(t0));
end
C = rmfield(C, 'Y');
save(fullfile(here, ['E2h_results' sfx '.mat']), 'C', 'R');
fprintf(['\nPure-delay nulls (E2g v2) for comparison: tau +12.8 ms insp 0.67/0.71, thirds P>A 0.00 0.05 0.61 A>P 0.53 0.21 0.00 (asym 1.06);\n' ...
         '                                          tau +32.7 ms insp 1.56/1.86, thirds P>A 0.00 0.07 1.45 A>P 1.27 0.56 0.00 (asym 1.19)\n']);
fprintf('Reading: the RC null whose inspiration PV, timing AND asymmetry match the real row is the transfer-free explanation of the real flux.\n');

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

function B = insp_pv(q, kmin, kei, dt)
% PV per breath during inspiration, both directions, from a (T-1) x 1 rate
nb = numel(kei);  B = nan(nb, 2);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    if numel(st) < 2; continue; end
    B(b,:) = [dt*sum(max(q(st), 0)), dt*sum(max(-q(st), 0))];
end
end

function [q, F, Bq] = run_quad(J, Y, Gc, o, sel, kmin, kei, dt, TVg)
[rho, m] = hk_core(-J, speye(size(J,1)), Y, Gc, o);
QG = hk_pendelluft(Gc, m, sel, dt);
F  = (diff(Gc.vol' * rho) / dt)';
a  = (F' * QG) / (F' * F);
q  = QG - a * F;
Bq = 100 * insp_pv(q, kmin, kei, dt) ./ TVg;
end

function TT = timing_thirds(q, kmin, kei, dt, TVg)
% PV in the early / mid / late third of each inspiration, both directions
nb = numel(kei);  X = nan(nb, 6);
for b = 1:nb
    st = kmin(b):kei(b)-1;  n = numel(st);
    if n < 3; continue; end
    u = ((0:n-1) + 0.5) / n;  bin = min(floor(3*u) + 1, 3);
    for h = 1:3
        qs = q(st(bin == h));
        X(b, h)   = 100 * dt * sum(max(qs, 0))  / TVg(b);
        X(b, 3+h) = 100 * dt * sum(max(-qs, 0)) / TVg(b);
    end
end
TT = median(X, 1, 'omitnan');
end

function [tau_ph, tau_xc] = regional_lag(a, b, fps, RR)
% tau > 0: a lags b. (1) phase of the respiratory fundamental, (2) peak of
% the whole-record cross-correlation with parabolic sub-frame interpolation
a = a(:)' - mean(a);  b = b(:)' - mean(b);  T = numel(a);
n  = 0:T-1;  w = 0.5 - 0.5*cos(2*pi*n/(T-1));
ea = exp(-1i*2*pi*RR*n/fps);
Xa = sum(a .* w .* ea);  Xb = sum(b .* w .* ea);
tau_ph = -angle(Xa / Xb) / (2*pi*RR);
L = 4;  r = zeros(1, 2*L+1);
for l = -L:L
    if l >= 0; i1 = 1+l:T; i2 = 1:T-l; else; i1 = 1:T+l; i2 = 1-l:T; end
    r(l+L+1) = (a(i1) * b(i2)') / (norm(a(i1)) * norm(b(i2)) + eps);
end
[~, k] = max(r);
if k > 1 && k < numel(r)
    d = 0.5 * (r(k-1) - r(k+1)) / (r(k-1) - 2*r(k) + r(k+1));
else
    d = 0;
end
tau_xc = ((k - L - 1) + d) / fps;
end

function s = ternary(c, a, b); if c; s = a; else; s = b; end; end

function y = rc_filter(x, fps, tau)
% first-order low-pass 1/(1 + i 2 pi f tau) in the frequency domain, applied
% to the detrended signal (linear trend kept), so that sub-frame tau works
x = x(:)';  T = numel(x);
tr = x(1) + (x(end) - x(1)) * (0:T-1) / (T-1);
f  = [0:floor(T/2), -ceil(T/2)+1:-1] * fps / T;
y  = real(ifft(fft(x - tr) ./ (1 + 1i*2*pi*f*tau))) + tr;
end

function B = exp_pv(q, kmin, kei, dt)
% PV per breath during EXPIRATION (end-inspiration -> next end-expiration)
nb = numel(kei);  B = nan(nb, 2);
for b = 1:nb
    st = kei(b):kmin(b+1)-1;
    if numel(st) < 2; continue; end
    B(b,:) = [dt*sum(max(q(st), 0)), dt*sum(max(-q(st), 0))];
end
end
