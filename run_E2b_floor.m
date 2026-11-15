%RUN_E2B_FLOOR  E2b: noise floor (null surrogates) and dose-response on the
%   REAL neonate background (injected pendelluft), for HK P1 and the image
%   estimators. Same data and pipeline as run_E2_neonate.
%
%   E2 gave HK post->ant 0.80 %TV, ant->post 2.28 %TV (CV ~0.7) and FRIC-vol
%   283 %TV (CV 0.07). Without a floor none of these is interpretable: the
%   FRIC step-sum accumulates frame-to-frame image noise over ~800 nodes.
%
%   Conditions (all reconstructed with the SAME hp, w, rho0 as the real data):
%     real          measured Y
%     null-w  r1,r2 zero pendelluft by construction: rho_null = w s(t)
%                   (every node follows the global profile s(t) with the
%                   P1 ventilation profile w), plus the real GN residual
%                   E = Y + Ac rgn after multivariate phase randomization
%                   (common random phases across channels: keeps the auto-
%                   and cross-spectra, destroys breath locking)
%     null-g  r1,r2 as null-w but the in-phase map has a +-30% anterior-
%                   posterior amplitude gradient that HK's w does NOT know
%                   (still zero pendelluft: one temporal profile everywhere)
%     inj d         Y + injected transient transfer in every inspiration:
%                   d % of that breath's GN TV moves post -> ant in the first
%                   half of the inspiration and back in the second half,
%                   d in {2.5, 5, 10, 20}. Expected increment for each HK
%                   direction: d %TV (slope 1).
%   Normalization: every estimator per breath divided by the REAL GN TV of
%   that breath (fixed denominator, so conditions are comparable).
%   Additional image estimator: FRIC-net = volume of nodes that LOST gas
%   between start and end of the inspiration (one difference, no step sum).
%
%   ~45 min (9 HK runs). Needs EIDORS and the HK-dEIT files.
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
MASK = 'thorax';   % 'lungs' reproduces the lung-mask log in README; 'thorax' (after E2i) writes *_thorax.mat
doses = [2.5 5 10 20];  grad_amp = 0.3;

% ---------------- data, filter, model (as run_E2_neonate) ----------------
D   = load_real_eit(fn);
fps = D.fps;  dt = 1/fps;  V = D.vv;  T = size(V, 2);
g   = sum(V ./ mean(V, 2) - 1, 1);
[S_, fr] = spectrum(g, fps);
RR  = peak_in(S_, fr, [0.4 1.6]);  HR = peak_in(S_, fr, [1.7 3.5]);
fc  = min(0.8*HR, max(2.2*RR, 0.5*(2*RR + 0.8*HR)));
V_lp = fft_lowpass(V, fps, fc, 0.2*fc);

fm = mk_library_model('neonate_16el_lungs');
order = electrode_order(fm, 'left');
fm.electrode = fm.electrode(order);  fm.stimulation = D.stim;  fm = mdl_normalize(fm, 1);
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
fprintf('RR %.2f Hz, HR %.2f Hz, fc %.2f Hz | %d breaths | hp %.3g\n', RR, HR, fc, nb, hp);

% geometry, P1 settings (fixed for every condition)
[ant_dir, c_t] = anterior_dir(fm);
yap   = (Gc.cent(:,1:2) - c_t) * ant_dir';      % anterior coordinate
ant_c = yap > 0;                                % anterior = dependent (prone)
VolA  = sum(Gc.vol(ant_c));  VolP = sum(Gc.vol(~ant_c));
L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/VolA - sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/VolP);
w_tidal = max(tivm, 0);
rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);

% ---------------- conditions ----------------
E   = Y + Ac * rgn;                             % GN residual (noise + model error)
s_w = gV / (Gc.vol' * w_tidal);                 % global profile for the w-map null
yg  = yap / max(abs(yap));                      % in [-1, 1], anterior > 0
w_g = w_tidal .* (1 + grad_amp * yg);
s_g = gV / (Gc.vol' * w_g);
pat = double(ant_c) / VolA - double(~ant_c) / VolP;   % unit transfer post -> ant
C = struct('name', {}, 'Y', {}, 'dose', {});
C(end+1) = struct('name', 'real', 'Y', Y, 'dose', 0);
C(end+1) = struct('name', 'null-w r1', 'Y', -Ac*(w_tidal*s_w) + phase_rand(E, 1), 'dose', NaN);
C(end+1) = struct('name', 'null-w r2', 'Y', -Ac*(w_tidal*s_w) + phase_rand(E, 2), 'dose', NaN);
C(end+1) = struct('name', 'null-g r1', 'Y', -Ac*(w_g*s_g) + phase_rand(E, 3), 'dose', NaN);
C(end+1) = struct('name', 'null-g r2', 'Y', -Ac*(w_g*s_g) + phase_rand(E, 4), 'dose', NaN);
for d = doses
    drho = zeros(Gc.N, T);
    for b = 1:nb
        ki = kmin(b):kei(b);  n = numel(ki);
        c  = sin(pi * (0:n-1) / (n-1));         % 0 -> 1 -> 0 over the inspiration
        drho(:, ki) = (d/100 * TVg(b)) * pat * c;
    end
    C(end+1) = struct('name', sprintf('inj %g', d), 'Y', Y - Ac*drho, 'dose', d); %#ok<SAGROW>
end

% ---------------- run ----------------
enames = {'HK post->ant', 'HK ant->post', 'FRIC-vol', 'FRIC-net', 'regout-ant', 'regout-post'};
Bp = cell(numel(C), 1);
for k = 1:numel(C)
    Yk  = C(k).Y;
    rgk = -gn_frames(Ac, Yk, hp);
    t0 = tic;
    [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
    B  = breath_metrics(m, rgk, Gc, ant_c, kmin, kei, dt);
    Bp{k} = 100 * B ./ TVg;
    act = (rho + rho0) <= 1e-3 * rho0;
    fprintf(['%-10s | misfit HK %.3f GN %.3f | bound %.3f | res %.0e %d it %.0f s | median %%TV: ' ...
             'HK P>A %.2f A>P %.2f | FRIC %.1f net %.2f | regout A %.2f P %.2f\n'], C(k).name, ...
             norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), norm(-Ac*rgk - Yk, 'fro')/norm(Yk, 'fro'), ...
             mean(act(:)), max(info.res), info.iters, toc(t0), median(Bp{k}, 1, 'omitnan'));
end
save(fullfile(here, ['E2b_results' sfx '.mat']), 'Bp', 'C', 'enames', 'doses', 'TVg', 'kmin', 'kei', 'P1', 'grad_amp');

% ---------------- summaries ----------------
med  = cell2mat(cellfun(@(x) median(x, 1, 'omitnan'), Bp, 'UniformOutput', false));   % cond x est
nul  = contains({C.name}, 'null');
inj  = ~isnan([C.dose]) & [C.dose] > 0;
fprintf('\n============ E2b: noise floor (max over the 4 null surrogates) vs real, %% of TV ============\n');
fprintf('%-14s %10s %10s %10s %12s\n', 'estimator', 'null-w max', 'null-g max', 'real', 'real/floor');
for j = 1:numel(enames)
    fw = max(med(contains({C.name}, 'null-w'), j));  fg = max(med(contains({C.name}, 'null-g'), j));
    fprintf('%-14s %10.2f %10.2f %10.2f %12.2f\n', enames{j}, fw, fg, med(1, j), med(1, j) / max(max(fw, fg), eps));
end
fprintf('\n============ E2b: dose-response on the real background (expected HK slope 1 per direction) ============\n');
fprintf('%-14s %8s %8s %8s   increments at doses %s\n', 'estimator', 'slope', 'icpt', 'R2', mat2str(doses));
dd = [0, doses];
for j = 1:numel(enames)
    inc = zeros(1, numel(dd));
    for q = 1:numel(doses)
        k = find(inj & [C.dose] == doses(q), 1);
        inc(q+1) = median(Bp{k}(:, j) - Bp{1}(:, j), 'omitnan');
    end
    pf = polyfit(dd, inc, 1);  res = inc - polyval(pf, dd);
    R2 = 1 - sum(res.^2) / max(sum((inc - mean(inc)).^2), eps);
    fprintf('%-14s %8.3f %8.3f %8.3f   %s\n', enames{j}, pf(1), pf(2), R2, mat2str(inc(2:end), 3));
end
fprintf('\nDetection dose (HK) = smallest d whose increment exceeds the null floor of that direction.\n');

% ======================================================================
function B = breath_metrics(m, rgn, Gc, sel, kmin, kei, dt)
nb = numel(kei);  B = nan(nb, 6);
QG = hk_pendelluft(Gc, m, sel, dt);
for b = 1:nb
    ki = kmin(b):kei(b);  st = ki(1):ki(end)-1;
    if numel(st) < 2; continue; end
    d  = diff(rgn(:, ki), 1, 2);
    dn = rgn(:, ki(end)) - rgn(:, ki(1));
    dA = Gc.vol(sel)' * d(sel, :);  dP = Gc.vol(~sel)' * d(~sel, :);
    B(b,:) = [dt*sum(max(QG(st), 0)), dt*sum(max(-QG(st), 0)), ...
              sum(sum(Gc.vol .* max(-d, 0))), sum(Gc.vol .* max(-dn, 0)), ...
              sum(max(-dA, 0)), sum(max(-dP, 0))];
end
end

function Es = phase_rand(E, seed)
% multivariate phase randomization: one random phase per frequency, shared
% by all channels (keeps auto- and cross-spectra)
rng(seed);
T = size(E, 2);  nh = floor((T-1)/2);
ph = exp(1i * 2*pi * rand(1, nh));
P  = ones(1, T);  P(2:nh+1) = ph;  P(T-nh+1:T) = conj(fliplr(ph));
Es = real(ifft(fft(E, [], 2) .* P, [], 2));
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

function [ant, c_t] = anterior_dir(fm)
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
c_t = wc(1:size(fm.elems,1));  c_l = wc(fm.mat_idx{2});  c_r = wc(fm.mat_idx{3});
rgt = (c_r - c_l) / norm(c_r - c_l);
ant = c_t - (c_l + c_r)/2;  ant = ant - dot(ant, rgt)*rgt;  ant = ant / norm(ant);
end

function order = electrode_order(fm, towards)
% same rule as run_E2_neonate/electrode_orientation
[ant, c_t] = anterior_dir(fm);
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
rgt = wc(fm.mat_idx{3}) - wc(fm.mat_idx{2});  rgt = rgt / norm(rgt);
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
