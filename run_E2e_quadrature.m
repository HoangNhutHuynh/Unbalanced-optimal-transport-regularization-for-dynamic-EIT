%RUN_E2E_QUADRATURE  E2e: lag-aware (quadrature) pendelluft estimate.
%
%   The dominant false-positive source on the real neonate data is an error in
%   the in-phase ventilation profile w: a +-30 % anterior-posterior amplitude
%   gradient unknown to HK gives 6.1 (P>A) / 7.4 (A>P) %TV of spurious flux
%   (E2c v2), while boundary-scaling mismatch (E2d) gives < 0.9 %TV and the
%   nuisance variants (E2c) move the estimate by < 1.5 %TV.
%   An amplitude error of w is IN PHASE with breathing: the spurious interface
%   flux is proportional to the global flow F(t) = d/dt (vol' rho), i.e.
%   QG_spur(t) = a F(t) with a constant a. Pendelluft is redistribution with a
%   regional TIME LAG, so it has a component orthogonal to F.
%   Quadrature estimator: fit a by least squares over the whole record,
%   r(t) = QG(t) - a F(t), PVq = dt sum_insp max(+-r, 0). a itself is reported
%   (fraction of the global flow crossing the interface, in-phase part).
%   Conditions (HK P1, no nuisance, as E2b): real, null-g+ / null-g- (+-30 %
%   gradient, zero pendelluft), real + injected transient transfer
%   d in {5, 10, 20} %TV. Reported: raw and quadrature PV per direction,
%   floors (max over the two nulls), dose-response slope/R2 for both.
%   Cost of the projection: part of a true pendelluft that is correlated with
%   F is removed too -> the quadrature slope is < the raw slope; the trade-off
%   is the floor reduction.
%
%   ~55 min (6 HK runs). Results in E2e_results.mat.
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
doses = [5 10 20];  grad_amp = 0.3;

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
yap = (Gc.cent(:,1:2) - c_t) * ant_dir';   yg = yap / max(abs(yap));
wgp = w_tidal .* (1 + grad_amp*yg);   wgm = w_tidal .* (1 - grad_amp*yg);
pat = double(ant_c) / VolA - double(~ant_c) / VolP;
C = struct('name', {}, 'Y', {}, 'dose', {});
C(end+1) = struct('name', 'real',    'Y', Y, 'dose', 0);
C(end+1) = struct('name', 'null-g+', 'Y', -Ac*(wgp*(gV/(Gc.vol'*wgp))) + phase_rand(E, 3), 'dose', NaN);
C(end+1) = struct('name', 'null-g-', 'Y', -Ac*(wgm*(gV/(Gc.vol'*wgm))) + phase_rand(E, 5), 'dose', NaN);
for d = doses
    drho = zeros(Gc.N, T);
    for b = 1:nb
        ki = kmin(b):kei(b);  n = numel(ki);
        drho(:, ki) = (d/100 * TVg(b)) * pat * sin(pi * (0:n-1) / (n-1));
    end
    C(end+1) = struct('name', sprintf('inj %g', d), 'Y', Y - Ac*drho, 'dose', d); %#ok<SAGROW>
end

% ---------------- runs ----------------
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
nC = numel(C);  Braw = cell(nC,1);  Bq = cell(nC,1);  aa = nan(nC,1);  QGs = cell(nC,1);  Fs = cell(nC,1);
for k = 1:nC
    t0 = tic;
    [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), C(k).Y, Gc, o);
    QG = hk_pendelluft(Gc, m, ant_c, dt);              % (T-1) x 1, rate into anterior
    F  = (diff(Gc.vol' * rho) / dt)';                  % (T-1) x 1, global flow rate
    aa(k) = (F' * QG) / (F' * F);
    r  = QG - aa(k) * F;
    Braw{k} = 100 * insp_pv(QG, kmin, kei, dt) ./ TVg;
    Bq{k}   = 100 * insp_pv(r,  kmin, kei, dt) ./ TVg;
    QGs{k} = QG;  Fs{k} = F;
    cc = corrcoef(QG, F);
    fprintf(['%-8s | a %+.4f corr(QG,F) %+.2f | raw P>A %.2f A>P %.2f | quadrature P>A %.2f A>P %.2f %%TV | ' ...
             'res %.0e %d it %.0f s\n'], C(k).name, aa(k), cc(1,2), median(Braw{k}, 1, 'omitnan'), ...
             median(Bq{k}, 1, 'omitnan'), max(info.res), info.iters, toc(t0));
end
save(fullfile(here, ['E2e_results' sfx '.mat']), 'C', 'Braw', 'Bq', 'aa', 'QGs', 'Fs', 'doses', 'TVg', 'kmin', 'kei', 'grad_amp');
C = rmfield(C, 'Y');

% ---------------- summaries ----------------
nm = {C.name};  isn = strncmp(nm, 'null', 4);
med = @(B) cell2mat(cellfun(@(x) median(x, 1, 'omitnan'), B, 'UniformOutput', false));
Mr = med(Braw);  Mq = med(Bq);
fprintf('\n============ E2e: real vs in-phase floor (max of null-g+/-), %%TV ============\n');
fprintf('%-12s %10s %10s %10s %10s\n', 'estimator', 'real P>A', 'floor P>A', 'real A>P', 'floor A>P');
fprintf('%-12s %10.2f %10.2f %10.2f %10.2f\n', 'raw', Mr(1,1), max(Mr(isn,1)), Mr(1,2), max(Mr(isn,2)));
fprintf('%-12s %10.2f %10.2f %10.2f %10.2f\n', 'quadrature', Mq(1,1), max(Mq(isn,1)), Mq(1,2), max(Mq(isn,2)));
fprintf('\n============ E2e: dose-response on the real background (increment vs real) ============\n');
fprintf('%-12s %-6s %8s %8s %8s   increments at %s\n', 'estimator', 'dir', 'slope', 'icpt', 'R2', mat2str(doses));
dd = [0 doses];
lab = {'raw', 'quadrature'};  BB = {Braw, Bq};  dn = {'P>A', 'A>P'};
for e = 1:2
    for j = 1:2
        inc = zeros(1, numel(dd));
        for q = 1:numel(doses)
            k = find([C.dose] == doses(q), 1);
            inc(q+1) = median(BB{e}{k}(:,j) - BB{e}{1}(:,j), 'omitnan');
        end
        pf = polyfit(dd, inc, 1);  rs = inc - polyval(pf, dd);
        R2 = 1 - sum(rs.^2) / max(sum((inc - mean(inc)).^2), eps);
        fprintf('%-12s %-6s %8.3f %8.3f %8.3f   %s\n', lab{e}, dn{j}, pf(1), pf(2), R2, mat2str(inc(2:end), 3));
    end
end
fprintf('\nDetection threshold of each estimator = its floor / its slope (in injected %%TV).\n');

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
