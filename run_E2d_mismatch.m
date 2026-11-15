%RUN_E2D_MISMATCH  E2d: model-mismatch nulls on the real neonate geometry.
%
%   E2b/E2c nulls were generated with the SAME forward operator as the
%   reconstruction (inverse crime). They have no anti-phase component
%   (element-GN anti-phase loss 6.9 %TV) whereas the real data have 51 %TV,
%   zero-lag, mirror image of inspiration (diag_E2_antiphase), which the
%   deflated nuisance of E2c v2 could not remove without removing gas
%   (cleaned loss still 48-50 %TV). Hypothesis: it is a LINEAR artefact of
%   the in-phase ventilation signal under forward-model mismatch (boundary
%   shape / size of the library neonate model vs this baby), so it lives in
%   the data image of ventilation itself.
%
%   Test: in-phase null (zero pendelluft: rho = w s(t), as null-w) generated
%   on a DEFORMED forward model and reconstructed with the nominal one.
%   Deformation = anisotropic scaling of all nodes about the thorax centre
%   along the anatomical axes (left-right factor sx, anterior-posterior sy);
%   mesh topology unchanged, so the same cluster map applies. Noise = real
%   GN residual, multivariate phase-randomized (as E2b).
%     levels (sx, sy): (1,1) control | (1.05,0.95) (1.10,0.90) wider/flatter
%                      | (0.95,1.05) (0.90,1.10) narrower/deeper
%   GN only, every level: element-GN anti-phase loss (%TV), sign consistency,
%   min gas per node / mean TIV (plausibility).
%   HK P1 (no nuisance), the 4 deformed levels: misfit, positivity bound,
%   TV HK/GN, P>A and A>P (%TV) = mismatch floor of the pendelluft estimate.
%   Real data reference (E2b): misfit 0.318, bound 0.052, P>A 0.70, A>P 2.03.
%
%   ~40 min (4 HK runs). Results in E2d_results.mat.
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
lev = [1 1; 1.05 0.95; 1.10 0.90; 0.95 1.05; 0.90 1.10];
hk_lev = 2:5;

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

% ---------------- mismatch nulls ----------------
R2 = [rgt_dir; ant_dir];                               % rows: L-R, A-P axes
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
Res = nan(size(lev,1), 9);
fprintf('\n');
for k = 1:size(lev, 1)
    fp = fm;
    q  = (fm.nodes(:,1:2) - c_t) * R2';
    q  = q .* lev(k,:);
    fp.nodes(:,1:2) = q * R2 + c_t;
    Jp = calc_jacobian(mk_image(fp, s0));
    Yk = -Jp * (Gc.EL * rho_null) + phase_rand(E, 10 + k);
    rgk = -gn_frames(Ac, Yk, hp);
    L   = apl(Yk);
    mng = min(rgk(:)) / tl;
    mis_gn = norm(-Ac*rgk - Yk, 'fro') / norm(Yk, 'fro');
    Res(k, 1:4) = [L, mng, mis_gn, norm(Jp - J, 'fro')/norm(J, 'fro')];
    line = sprintf('(sx %.2f, sy %.2f) | ||Jp-J||/||J|| %.3f | element-GN anti-phase loss %5.1f %%TV | GN min gas / mean TIV %6.1f | GN misfit %.3f', ...
                   lev(k,1), lev(k,2), Res(k,4), L, mng, mis_gn);
    if ismember(k, hk_lev)
        t0 = tic;
        [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
        Bp  = 100 * breath_metrics(m, Gc, ant_c, kmin, kei, dt) ./ TVg;
        hV  = Gc.vol' * rho;
        TVh = arrayfun(@(b) hV(kei(b)) - hV(kmin(b)), 1:nb)';
        act = (rho + rho0) <= 1e-3 * rho0;
        Res(k, 5:9) = [norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), mean(act(:)), median(TVh ./ TVg), ...
                       median(Bp(:,1), 'omitnan'), median(Bp(:,2), 'omitnan')];
        line = [line, sprintf(' || HK misfit %.3f bound %.3f TV HK/GN %.2f | P>A %.2f A>P %.2f %%TV | res %.0e %d it %.0f s', ...
                Res(k,5:9), max(info.res), info.iters, toc(t0))];
    end
    fprintf('%s\n', line);
end
save(fullfile(here, 'E2d_results.mat'), 'Res', 'lev', 'hk_lev', 'kmin', 'kei', 'TVg');
fprintf('\nreal-data reference: anti-phase 51.0 %%TV | HK misfit 0.318 bound 0.052 | P>A 0.70 A>P 2.03 %%TV (E2b)\n');
fprintf('in-phase floors without mismatch (E2c v2, g 3): P>A 6.13 A>P 7.38 %%TV for +-30 %% amplitude gradient\n');

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
