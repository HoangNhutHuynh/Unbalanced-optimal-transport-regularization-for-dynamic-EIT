%RUN_E2F_FLOORS  E2f: all false-positive floors of the QUADRATURE estimator.
%
%   E2e: the in-phase/quadrature split removes the w-amplitude floor
%   (null-g+/-: corr(QG,F) = +-1.00; floor 6.3/7.3 -> 0.06-0.08 %TV) with no
%   loss of sensitivity (slope 0.77/0.70, R2 0.99). The real data give
%   quadrature P>A 0.76, A>P 1.73 %TV, i.e. 10-20x the w-floor. Before this
%   can be read as pendelluft every other floor must be measured with the
%   same estimator:
%     null-w r1, r2   noise only (correct w, real residual phase-randomized)
%     mm (1.10,0.90), mm (0.90,1.10)  model mismatch nulls of E2d
%     real + nuisance (deflated, g 3, as E2c v2)  robustness of the real value
%   Plus a per-breath empirical test: for each direction, the fraction of the
%   pooled null breaths (null-w and mismatch) whose quadrature PV is >= the
%   real per-breath median (one-sided empirical p), and the number of real
%   breaths above the pooled null 95th percentile.
%
%   ~1 h (5 HK runs). Results in E2f_results.mat.
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
K1 = 40;  g_nuis = 3;

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
R2  = [rgt_dir; ant_dir];
C = struct('name', {}, 'Y', {}, 'nuis', {});
C(end+1) = struct('name', 'real',      'Y', Y, 'nuis', false);
C(end+1) = struct('name', 'null-w r1', 'Y', -Ac*rho_null + phase_rand(E, 1), 'nuis', false);
C(end+1) = struct('name', 'null-w r2', 'Y', -Ac*rho_null + phase_rand(E, 2), 'nuis', false);
for lv = [1.10 0.90; 0.90 1.10]'
    fp = fm;  q = ((fm.nodes(:,1:2) - c_t) * R2') .* lv';  fp.nodes(:,1:2) = q * R2 + c_t;
    Jp = calc_jacobian(mk_image(fp, s0));
    C(end+1) = struct('name', sprintf('mm %.2f/%.2f', lv), 'Y', -Jp*(Gc.EL*rho_null) + phase_rand(E, 20 + round(100*lv(1))), 'nuis', false); %#ok<SAGROW>
end
C(end+1) = struct('name', 'real+nuis', 'Y', Y, 'nuis', true);

% deflated nuisance operator (as run_E2c_nuisance v2)
c0  = median(sqrt(sum(Ac.^2, 1)));
cn  = @(X) X ./ max(sqrt(sum(X.^2, 1)), eps) * c0;
eta_op = cn(movement_jac(fm, s0, [1 2], 1e-6));
if strcmp(MASK, 'lungs')       % whole thorax: no tissue outside the domain, electrode movement only
    Gn = calc_dual_graph(fm, setdiff((1:Ne)', lung_idx), struct('target_N', K1));
    eta_op = [cn(J * Gn.EL), eta_op];
end
yap = (Gc.cent(:,1:2) - c_t) * ant_dir';   yg = yap / max(abs(yap));
xlr = (Gc.cent(:,1:2) - c_t) * rgt_dir';   xg = xlr / max(abs(xlr));
isLc = (Gc.EL' * double(grp == 1)) ./ max(full(sum(Gc.EL, 1))', 1) > 0.5;
[Uv, Sv, ~] = svd(Ac * [w_tidal, w_tidal.*yg, w_tidal.*xg, w_tidal.*isLc], 'econ');
Uv = Uv(:, diag(Sv) > 1e-8*Sv(1));
eta_op = eta_op - Uv * (Uv' * eta_op);
c0n = c0 / normest(J, 1e-4);

% ---------------- image baselines (GN, same breaths / split / nulls; seconds) ----------------
BASE_ONLY = false;    % true: baselines only, then stop; false: also the HK runs (~1 h, E2f log)
isnB = strncmp({C.name}, 'null', 4) | strncmp({C.name}, 'mm', 2);
Bb = cell(numel(C), 1);  lagB = nan(numel(C), 1);
for k = 1:numel(C)
    [Bb{k}, lagB(k), nmB] = pend_estimators(-gn_frames(Ac, C(k).Y, hp), Gc.vol, ant_c, kmin, kei, dt);
end
poolB = cell2mat(Bb(isnB));
fprintf('\n============ E2f baselines on GN images (%%TV per breath; lag of posterior behind anterior) ============\n');
fprintf('%-8s | %7s | %7s %7s | %s\n', 'estim.', 'real', 'null md', 'null p95', 'real breaths > p95 | real/p95');
for j = 1:numel(nmB)
    x = Bb{1}(:, j);  p95 = prc(poolB(:, j), 95);
    fprintf('%-8s | %7.2f | %7.2f %7.2f | %d of %d | %.2f\n', nmB{j}, median(x, 'omitnan'), ...
            median(poolB(:, j), 'omitnan'), p95, nnz(x > p95), nnz(isfinite(x)), median(x, 'omitnan') / max(p95, eps));
end
fprintf('lag (ms): real %+.1f | nulls %s\n', lagB(1), mat2str(round(lagB(isnB)', 1)));
save(fullfile(here, ['E2f_baselines' sfx '.mat']), 'Bb', 'lagB', 'nmB', 'isnB', 'kmin', 'kei');
if BASE_ONLY; return; end

% ---------------- runs ----------------
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
on = o;  on.use_eta = true;  on.eta_op = eta_op;  on.gamma = g_nuis * c0n^2;
nC = numel(C);  Braw = cell(nC,1);  Bq = cell(nC,1);  aa = nan(nC,1);
for k = 1:nC
    t0 = tic;
    if C(k).nuis; oo = on; else; oo = o; end
    [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), C(k).Y, Gc, oo);
    QG = hk_pendelluft(Gc, m, ant_c, dt);
    F  = (diff(Gc.vol' * rho) / dt)';
    aa(k) = (F' * QG) / (F' * F);
    r  = QG - aa(k) * F;
    Braw{k} = 100 * insp_pv(QG, kmin, kei, dt) ./ TVg;
    Bq{k}   = 100 * insp_pv(r,  kmin, kei, dt) ./ TVg;
    cc = corrcoef(QG, F);
    fprintf(['%-12s | a %+.4f corr(QG,F) %+.2f | raw P>A %.2f A>P %.2f | quadrature P>A %.2f A>P %.2f %%TV | ' ...
             'res %.0e %d it %.0f s\n'], C(k).name, aa(k), cc(1,2), median(Braw{k}, 1, 'omitnan'), ...
             median(Bq{k}, 1, 'omitnan'), max(info.res), info.iters, toc(t0));
end
C = rmfield(C, 'Y');
save(fullfile(here, ['E2f_results' sfx '.mat']), 'C', 'Braw', 'Bq', 'aa', 'TVg', 'kmin', 'kei', 'g_nuis');

% ---------------- floors and per-breath test ----------------
nm = {C.name};
isnull = strncmp(nm, 'null', 4) | strncmp(nm, 'mm', 2);
med = @(B) cell2mat(cellfun(@(x) median(x, 1, 'omitnan'), B, 'UniformOutput', false));
Mq = med(Bq);  Mr = med(Braw);
fprintf('\n============ E2f: quadrature floors (median %%TV per condition) ============\n');
fprintf('%-12s %8s %8s   (raw %8s %8s)\n', 'condition', 'P>A', 'A>P', 'P>A', 'A>P');
for k = 1:nC; fprintf('%-12s %8.2f %8.2f   (raw %8.2f %8.2f)\n', nm{k}, Mq(k,:), Mr(k,:)); end
fprintf('w-amplitude floor (E2e, null-g+/-): quadrature 0.08 / 0.08, raw 6.29 / 7.26\n');
pool = cell2mat(Bq(isnull));
fprintf('\n============ E2f: per-breath test, real (no nuisance) vs pooled nulls (%d null breaths) ============\n', size(pool,1));
dn = {'P>A', 'A>P'};
for j = 1:2
    x = Bq{1}(:, j);  x = x(isfinite(x));  pj = pool(:, j);  pj = pj(isfinite(pj));
    p95 = prc(pj, 95);
    fprintf('%s: real median %.2f | null median %.2f, 95th pct %.2f, max %.2f | empirical p (null >= real median) %.3f | real breaths > null p95: %d of %d\n', ...
            dn{j}, median(x), median(pj), p95, max(pj), mean(pj >= median(x)), nnz(x > p95), numel(x));
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


function q = prc(x, pc)
x = sort(x(:));  n = numel(x);  r = 1 + (n-1)*pc/100;  lo = floor(r);  hi = ceil(r);
q = x(lo) + (r - lo) * (x(hi) - x(lo));
end
