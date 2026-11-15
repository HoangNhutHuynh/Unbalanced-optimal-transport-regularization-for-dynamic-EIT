%RUN_E2I_MASK  E2i: robustness of the quadrature HK estimate to the lung mask.
%
%   The lung-only reconstructions (HK, cluster GN) assume that gas changes
%   only inside the library neonate lungs. The whole-domain element GN image
%   (make_figs_HK, Fig02) puts much of the ventilation centrally, between the
%   model lungs, with a negative rim near the chest wall: either the central
%   bias of the NOSER-weighted GN, or a mismatch between the library lung mask
%   and this prone baby's lungs. E2d tested boundary scaling only.
%   Configurations (same data, breaths, P1 settings, quadrature estimator):
%     M1  lungs dilated by 2 element layers (shared-node neighbours)
%     M2  lungs dilated by 4 element layers
%     M3  whole thorax (every element)
%   added tissue is a separate cluster group (clusters never mix lung and
%   tissue); target_N scales with the domain volume (800 for the lungs).
%   For each configuration: real data and an in-phase null (w of that
%   configuration, real residual phase-randomized). Reported: GN/HK misfit,
%   positivity-bound fraction, TV HK/GN, share of the HK tidal volume placed
%   OUTSIDE the anatomical lungs, coefficient a and corr(QG, F), quadrature PV
%   per direction (% of that configuration's GN TV), asymmetry, timing thirds.
%   Reference M0 (nominal lungs, E2e/E2f): real 0.76 / 1.73 %TV, null-w
%   0.02-0.06 %TV, timing P>A 0 0 0.57 | A>P 0.86 0.56 0, asymmetry 2.27.
%
%   ~1 h (6 HK runs). Results in E2i_results.mat.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
fn = fullfile(here, 'Data', 'baby', 'if-neonate-spontaneous (1)', 'P04P-1016.get');
P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_lung = 800;  N_max = 1600;  tau0 = 100;  it_hk = 6000;

% ---------------- data and model (as run_E2_neonate) ----------------
D   = load_real_eit(fn);  fps = D.fps;  dt = 1/fps;  V = D.vv;  T = size(V, 2);
g   = sum(V ./ mean(V, 2) - 1, 1);
[S_, fr] = spectrum(g, fps);
RR  = peak_in(S_, fr, [0.4 1.6]);  HR = peak_in(S_, fr, [1.7 3.5]);
fc  = min(0.8*HR, max(2.2*RR, 0.5*(2*RR + 0.8*HR)));
V_lp = fft_lowpass(V, fps, fc, 0.2*fc);
fm = mk_library_model('neonate_16el_lungs');
fm.electrode = fm.electrode(electrode_order(fm, 'left'));
fm.stimulation = D.stim;  fm = mdl_normalize(fm, 1);
Ne = size(fm.elems, 1);
isL = false(Ne,1);  isL(fm.mat_idx{2}) = true;  isR = false(Ne,1);  isR(fm.mat_idx{3}) = true;
isLung = isL | isR;
s0 = ones(Ne,1);  s0(isLung) = sig_lung;
J  = calc_jacobian(mk_image(fm, s0));
S2 = load(fullfile(here, 'E2_results.mat'));  kmin = S2.kmin(:)';  kei = S2.kei(:)';  nb = numel(kei);
Y  = calc_difference_data(mean(V_lp(:, kmin), 2), V_lp, fm);
[ant, ~, c_t] = axes_dir(fm);
vol = get_elem_volume(fm);

% ---------------- masks ----------------
E2N  = sparse(repmat((1:Ne)', 1, size(fm.elems, 2)), fm.elems, 1, Ne, size(fm.nodes, 1));
grow = @(x) (E2N * (E2N' * double(x))) > 0;
m1 = isLung;  for i = 1:2; m1 = grow(m1); end
m2 = isLung;  for i = 1:4; m2 = grow(m2); end
cfg = struct('name', {'M1 lungs +2 layers', 'M2 lungs +4 layers', 'M3 whole thorax'}, 'mask', {m1, m2, true(Ne,1)});
for c = 1:numel(cfg)
    fprintf('%s: %.0f %% of the thorax volume (lungs %.0f %%)\n', cfg(c).name, ...
            100*sum(vol(cfg(c).mask))/sum(vol), 100*sum(vol(isLung))/sum(vol));
end

% ---------------- runs ----------------
R = struct('cfg', {}, 'cond', {}, 'Bq', {}, 'TT', {}, 'a', {}, 'cc', {}, 'mis', {}, 'misGN', {}, ...
           'bound', {}, 'tvr', {}, 'outside', {});
fprintf('\n');
for c = 1:numel(cfg)
    idx = find(cfg(c).mask);
    grp = 3 * ones(Ne, 1);  grp(isL) = 1;  grp(isR) = 2;
    Nc  = min(N_max, round(N_lung * sum(vol(idx)) / sum(vol(isLung))));
    Gc  = calc_dual_graph(fm, idx, struct('target_N', Nc, 'groups', grp));
    Ac  = J * Gc.EL;
    hp  = lcurve_hp(Ac, Y);
    rgn = -gn_frames(Ac, Y, hp);
    gV  = Gc.vol' * rgn;
    TVg = arrayfun(@(b) gV(kei(b)) - gV(kmin(b)), (1:nb)');
    tivm = mean(rgn(:, kei) - rgn(:, kmin(1:nb)), 2);
    w    = max(tivm, 0);
    rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);
    ant_c = (Gc.cent(:,1:2) - c_t) * ant' > 0;
    L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/sum(Gc.vol(ant_c)) - ...
                 sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/sum(Gc.vol(~ant_c)));
    inlung = (Gc.EL' * double(isLung)) ./ max(full(sum(Gc.EL, 1))', 1) > 0.5;
    E    = Y + Ac * rgn;
    Ynul = -Ac * (w * (gV / (Gc.vol' * w))) + phase_rand(E, 1);
    o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
               'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
               'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w);
    fprintf('%s: %d clusters (%d lung), GN hp %.3g, GN TV / nominal-mask GN TV %.2f, GN TIV share outside lungs %.2f\n', ...
            cfg(c).name, Gc.N, nnz(inlung), hp, median(TVg) / median(S2.M(:,1)), ...
            sum(Gc.vol(~inlung) .* tivm(~inlung)) / sum(Gc.vol .* tivm));
    for k = 1:2
        if k == 1; Yk = Y; nm = 'real'; else; Yk = Ynul; nm = 'null-w'; end
        t0 = tic;
        [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
        QG = hk_pendelluft(Gc, m, ant_c, dt);
        F  = (diff(Gc.vol' * rho) / dt)';
        a  = (F' * QG) / (F' * F);  r = QG - a * F;  cc = corrcoef(QG, F);
        Bq = 100 * insp_pv(r, kmin, kei, dt) ./ TVg;
        TT = timing_thirds(r, kmin, kei, dt, TVg);
        rk = -gn_frames(Ac, Yk, hp);
        hV = Gc.vol' * rho;
        TVh = arrayfun(@(b) hV(kei(b)) - hV(kmin(b)), (1:nb)');
        tivh = mean(rho(:, kei) - rho(:, kmin(1:nb)), 2);
        act = (rho + rho0) <= 1e-3 * rho0;
        R(end+1) = struct('cfg', cfg(c).name, 'cond', nm, 'Bq', Bq, 'TT', TT, 'a', a, 'cc', cc(1,2), ...
                          'mis', norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), 'misGN', norm(-Ac*rk - Yk, 'fro')/norm(Yk, 'fro'), ...
                          'bound', mean(act(:)), 'tvr', median(TVh ./ TVg), ...
                          'outside', sum(Gc.vol(~inlung) .* tivh(~inlung)) / sum(Gc.vol .* tivh)); %#ok<SAGROW>
        mq = median(Bq, 1, 'omitnan');
        fprintf(['  %-6s | misfit HK %.3f GN %.3f | bound %.3f | TV HK/GN %.2f | HK TIV outside lungs %.2f | a %+.4f corr %+.2f | ' ...
                 'quadrature P>A %.2f A>P %.2f %%TV (asym %.2f) | thirds P>A %.2f %.2f %.2f A>P %.2f %.2f %.2f | res %.0e %d it %.0f s\n'], ...
                 nm, R(end).mis, R(end).misGN, R(end).bound, R(end).tvr, R(end).outside, a, cc(1,2), mq, ...
                 mq(2)/max(mq(1), eps), TT, max(info.res), info.iters, toc(t0));
    end
end
save(fullfile(here, 'E2i_results.mat'), 'R', 'cfg', 'kmin', 'kei');

fprintf('\n============ E2i: quadrature PV vs lung mask (median %%TV of that configuration''s GN TV) ============\n');
fprintf('%-20s %10s %10s %10s %10s %8s\n', 'configuration', 'real P>A', 'real A>P', 'null P>A', 'null A>P', 'asym');
fprintf('%-20s %10.2f %10.2f %10s %10s %8.2f   (E2e/E2f)\n', 'M0 nominal lungs', 0.76, 1.73, '0.02-0.05', '0.05-0.06', 2.27);
for c = 1:numel(cfg)
    a1 = R(strcmp({R.cfg}, cfg(c).name) & strcmp({R.cond}, 'real'));
    a0 = R(strcmp({R.cfg}, cfg(c).name) & strcmp({R.cond}, 'null-w'));
    mr = median(a1.Bq, 1, 'omitnan');  mn = median(a0.Bq, 1, 'omitnan');
    fprintf('%-20s %10.2f %10.2f %10.2f %10.2f %8.2f\n', cfg(c).name, mr, mn, mr(2)/max(mr(1), eps));
end
fprintf('Reading: robust if real >> null in every configuration and real PV / timing / asymmetry stay close to M0.\n');

% ======================================================================
function Es = phase_rand(E, seed)
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

