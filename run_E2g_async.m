%RUN_E2G_ASYNC  E2g: is the quadrature HK flux explained by regional asynchrony?
%
%   E2f: every false-positive floor of the quadrature estimator is <= 0.08 %TV
%   (noise, model mismatch, w amplitude); the real data give P>A 0.76, A>P
%   1.73 %TV (0.74 / 1.43 with the nuisance), 8 of 9 breaths above the pooled
%   null 95th percentile in both directions.
%   Remaining confound (the identifiability limit of difference EIT): two
%   regions that fill with a TIME LAG from the airway, without exchanging gas,
%   give the same regional curves as a lagged inter-regional transfer. HK
%   reports the minimal-action redistribution consistent with the curves, so
%   asynchrony maps to quadrature PV. This script calibrates that map and
%   measures the real lag:
%     1. real regional lag between the anterior (dependent, prone) and the
%        posterior GN gas curves: phase of the respiratory fundamental and
%        whole-record cross-correlation with parabolic sub-frame peak
%        (tau > 0: anterior LAGS posterior); same on an in-phase null (bias)
%     2. timing of the real quadrature flux within inspiration (thirds)
%     3. asynchrony nulls (zero transfer by construction): anterior part of
%        the in-phase map delayed by tau, rho = w_P s(t) + w_A s(t - tau),
%        tau in {-2, -1, +1, +2} x |tau_real| (or +-50/100 ms if tau_real ~ 0),
%        real residual phase-randomized; quadrature PV vs tau.
%   Reading: if PV(tau_real) on the asynchrony curve ~ the real PV, the HK
%   estimate is fully accounted for by the measured asynchrony (it is then
%   its transfer-equivalent volume); if the real PV is much larger, the
%   regional curves contain redistribution beyond a pure delay.
%
%   v2: tau at +-|tau_phase|, +-|tau_xcorr| (v1 +-50/100 ms kept as 'prev'),
%   timing of every null, both lag estimates. ~45 min (5 HK runs). E2g_v2_results.mat.
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

% ---------------- 1. regional lag, real and in-phase null ----------------
VA = Gc.vol(ant_c)'  * rgn(ant_c, :);   VP = Gc.vol(~ant_c)' * rgn(~ant_c, :);
[tau_ph, tau_xc] = regional_lag(VA, VP, fps, RR);
rgN = -gn_frames(Ac, -Ac*rho_null + phase_rand(E, 1), hp);
[tn_ph, tn_xc] = regional_lag(Gc.vol(ant_c)'*rgN(ant_c,:), Gc.vol(~ant_c)'*rgN(~ant_c,:), fps, RR);
fprintf('regional lag anterior vs posterior (GN, > 0: anterior lags): real %+.1f ms (phase) %+.1f ms (xcorr) | in-phase null %+.1f / %+.1f ms\n', ...
        1e3*tau_ph, 1e3*tau_xc, 1e3*tn_ph, 1e3*tn_xc);
fprintf('  (frame interval %.1f ms, inspiration median %.0f ms)\n', 1e3*dt, 1e3*median(kei(:) - kmin(1:nb)')*dt);

% ---------------- HK on real: timing of the quadrature flux ----------------
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
[q_real, F_real, Bq_real] = run_quad(J, Y, Gc, o, ant_c, kmin, kei, dt, TVg);
TT = timing_thirds(q_real, kmin, kei, dt, TVg);
fprintf('real quadrature PV %%TV: P>A %.2f A>P %.2f | by thirds of inspiration (median over breaths):\n', median(Bq_real, 1, 'omitnan'));
fprintf('  P>A early %.2f mid %.2f late %.2f | A>P early %.2f mid %.2f late %.2f\n', TT);

% ---------------- 3. asynchrony nulls ----------------
tref = max(abs(tau_ph), abs(tau_xc));
% v2: v1 ran tau = +-50, +-100 ms (Steve, 2026-11-04; kept in 'prev' below);
% v2 fills the curve at the two measured lag magnitudes and records timing
taus = sort(reshape([-1; 1] * [abs(tau_ph) abs(tau_xc)], 1, []));
prev = [-0.0807 4.42 6.27; -0.0551 2.48 3.00; 0.0470 2.16 2.84; 0.0873 3.70 5.65];   % [GN lag, P>A, A>P]
if ~strcmp(MASK, 'lungs')      % v1 points are lung-mask numbers: run +-50, +-100 ms on this domain instead
    taus = sort([taus, -0.10, -0.05, 0.05, 0.10]);  prev = zeros(0, 3);
end
s_glob = gV / (Gc.vol' * w_tidal);
t  = (0:T-1) * dt;
wA = w_tidal .* ant_c;   wP = w_tidal .* ~ant_c;
Res = nan(numel(taus), 4);  TTn = nan(numel(taus), 6);
for k = 1:numel(taus)
    s_d = interp1(t, s_glob, min(max(t - taus(k), 0), t(end)), 'pchip');
    rho_as = wP * s_glob + wA * s_d;
    Yk = -Ac * rho_as + phase_rand(E, 40 + k);
    [qk, ~, Bq] = run_quad(J, Yk, Gc, o, ant_c, kmin, kei, dt, TVg);
    TTn(k,:) = timing_thirds(qk, kmin, kei, dt, TVg);
    rgk = -gn_frames(Ac, Yk, hp);
    [tk, ~] = regional_lag(Gc.vol(ant_c)'*rgk(ant_c,:), Gc.vol(~ant_c)'*rgk(~ant_c,:), fps, RR);
    Res(k,:) = [taus(k), tk, median(Bq, 1, 'omitnan')];
    fprintf(['asynchrony null tau %+6.1f ms (GN-measured %+6.1f ms) | quadrature P>A %.2f A>P %.2f %%TV | ' ...
             'thirds P>A %.2f %.2f %.2f  A>P %.2f %.2f %.2f\n'], 1e3*taus(k), 1e3*tk, Res(k,3:4), TTn(k,:));
end
save(fullfile(here, ['E2g_v2_results' sfx '.mat']), 'tau_ph', 'tau_xc', 'tn_ph', 'tn_xc', 'Bq_real', 'TT', 'Res', 'TTn', 'taus', 'prev');
ResAll = [Res(:, 2:4); prev];

% interpolate the asynchrony curve (v1 + v2 points, GN-measured lags) at both lag estimates
fprintf('\n============ E2g: real PV vs asynchrony-equivalent PV at the measured lag ============\n');
fprintf('asynchrony curve points (GN lag ms: P>A, A>P):');
[~, io] = sort(ResAll(:,1));
fprintf(' [%+.0f: %.2f, %.2f]', [1e3*ResAll(io,1), ResAll(io,2:3)]');  fprintf('\n');
for tr = [tau_ph tau_xc]
    for j = 1:2
        [ts, ix] = sort([ResAll(:,1); 0]);  pv = [ResAll(:, 1+j); 0];  pv = pv(ix);
        [ts, iu] = unique(ts);  pv = pv(iu);
        eq = interp1(ts, pv, tr, 'linear', 'extrap');
        fprintf('%s: real %.2f %%TV | asynchrony curve at tau %+.1f ms: %.2f %%TV | ratio real/asynchrony %.2f\n', ...
                ternary(j == 1, 'P>A', 'A>P'), median(Bq_real(:,j), 'omitnan'), 1e3*tr, eq, ...
                median(Bq_real(:,j), 'omitnan') / max(eq, eps));
    end
end
fprintf('real timing (thirds): P>A %.2f %.2f %.2f  A>P %.2f %.2f %.2f -> compare with the null rows of each sign\n', TT);
fprintf('(interpolation uses the GN-measured lag of each null, plus the point (0, 0))\n');

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
