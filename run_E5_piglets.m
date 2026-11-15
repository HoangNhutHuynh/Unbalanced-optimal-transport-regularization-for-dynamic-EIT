%RUN_E5_PIGLETS  E5: quadrature HK asynchrony index during a stepwise vital
%   capacity manoeuvre in newborn piglets: normal, lavaged and surfactant-
%   treated lung (2 animals x 3 lung states).
%
%   Data: EIDORS contrib "Dargaville-ICM2010" (Data\Pig_torso): Dargaville PA,
%   Rimensberger PC, Frerichs I, Intensive Care Med 36:1953-1961, 2010 (cite;
%   acknowledge I. Frerichs). Protocol (paper, Methods): newborn piglets
%   (2.3 +- 0.28 kg), SUPINE, anaesthetised and paralysed (pancuronium) ->
%   passive ventilation; pressure control, inflating pressure 10 cmH2O above
%   PEEP, inspiratory time 0.65 s, RR 20 /min; after disconnection to ambient
%   pressure, PEEP 5 -> 25 cmH2O in 5-cmH2O steps and back to 0, ~15 s
%   (>= 4 breaths) per level; Goe-MF II, 16 electrodes at the 6th parasternal
%   intercostal space, 13 Hz, 180 s (2340 frames) per manoeuvre. Electrode 1
%   anterior, numbered clockwise: 5 right lateral, 9 posterior, 13 left.
%   Files: g08-c003 / g10-c005 normal, g08-c017 / g10-c019 lavaged,
%   g08-c035 / g10-c029 surfactant.
%   ASSUMPTION: no piglet thorax with left/right lungs exists in the EIDORS
%   library (pig_23kg has a single lung contour), so the neonate thorax is
%   used on the WHOLE-THORAX domain (E2i: the lung mask matters more than the
%   outline). Supine -> the POSTERIOR (dorsal) region is dependent; P>A =
%   dorsal -> ventral.
%
%   Pipeline per manoeuvre (whole 180 s in one HK run):
%     breaths on the detrended GN gas volume (EELV steps removed by a moving
%     median); reference = end-expiration of the 3 lowest-EELV breaths (so the
%     gas content stays >= 0 through the manoeuvre); PEEP segmentation of the
%     breath-wise EELV by optimal piecewise-constant fit (dynamic programming)
%     into 11 levels labelled [0 5 10 15 20 25 20 15 10 5 0] - printed with the
%     segment EELV so the labelling can be checked; first breath after each
%     step change discarded; HK P1 (no nuisance), quadrature split r = QG - aF;
%     per PEEP level: median quadrature PV per direction (% of GN TV), timing
%     thirds, lag index L (> 0: dorsal later), GN TV, EELV.
%   Hypotheses: H1 above the in-phase null; H2 lavage raises and surfactant
%   lowers the index; H3 L depends on PEEP (patient 1 of E3: sign change at
%   18-20 cmH2O) and differs between inflation and deflation limbs.
%   In-phase nulls (whole manoeuvre) for the files in RUN_NULL.
%   RESUMABLE (E5_results\<file>.mat).  ~1.5-2 h per manoeuvre.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
ddir   = fullfile(here, 'Data', 'Pig_torso');
outdir = fullfile(here, 'E5_results');  if ~exist(outdir, 'dir'); mkdir(outdir); end
MODEL = 'neonate_16el_lungs';  TOWARDS = 'right';      % clockwise: e5 right lateral
F = {'g08-c003', 'g08', 'normal';  'g08-c017', 'g08', 'lavaged';  'g08-c035', 'g08', 'surfactant';
     'g10-c005', 'g10', 'normal';  'g10-c019', 'g10', 'lavaged';  'g10-c029', 'g10', 'surfactant'};
RUN_NULL = {'g08-c003', 'g08-c017'};
PEEPSEQ  = [0 5 10 15 20 25 20 15 10 5 0];
LIMB     = {'0', 'inc', 'inc', 'inc', 'inc', 'top', 'dec', 'dec', 'dec', 'dec', 'end'};
P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_c = 800;  tau0 = 100;  it_hk = 6000;

% ---------------- model (once) ----------------
D0 = load_real_eit(fullfile(ddir, [F{1,1} '.get']));
fm = mk_library_model(MODEL);
fm.electrode = fm.electrode(electrode_order(fm, TOWARDS));
fm.stimulation = D0.stim;  fm = mdl_normalize(fm, 1);
Ne = size(fm.elems, 1);
isL = false(Ne,1);  isL(fm.mat_idx{2}) = true;  isR = false(Ne,1);  isR(fm.mat_idx{3}) = true;
s0 = ones(Ne,1);  s0(isL | isR) = sig_lung;
J  = calc_jacobian(mk_image(fm, s0));
grp = 3 * ones(Ne,1);  grp(isL) = 1;  grp(isR) = 2;
Gc = calc_dual_graph(fm, (1:Ne)', struct('target_N', N_c, 'groups', grp));
Ac = J * Gc.EL;
[ant, ~, c_t] = axes_dir(fm);
ant_c = (Gc.cent(:,1:2) - c_t) * ant' > 0;
L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/sum(Gc.vol(ant_c)) - ...
             sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/sum(Gc.vol(~ant_c)));
fprintf('model %s (electrodes towards the %s), whole thorax: %d clusters\n', MODEL, TOWARDS, Gc.N);

for f = 1:size(F, 1)
    id = F{f,1};
    rfile = fullfile(outdir, [id '.mat']);
    if exist(rfile, 'file'); fprintf('%s: done (skipped)\n', id); continue; end
    t_file = tic;
    D = load_real_eit(fullfile(ddir, [id '.get']));
    fps = D.fps;  dt = 1/fps;  V = D.vv;  T = size(V, 2);
    fprintf('\n%s (%s, %s): %d frames at %g fps (%s) | bad %d, flagged ch %d, jumps %d\n', id, F{f,2}, F{f,3}, ...
            T, fps, D.fps_source, numel(D.qc.bad_frames), numel(D.qc.chan_flag), numel(D.qc.jump_frames));
    % ---- filter ----
    g = sum(V ./ mean(V, 2) - 1, 1);
    g = g - movmedian(g, round(6*fps));                  % remove the EELV steps before the spectrum
    [S_, frq] = spectrum(g, fps);
    RR = peak_in(S_, frq, [0.15 1.0]);
    hb = frq >= max(1.6*RR, 0.75) & frq <= min(3.5, 0.45*fps);
    [pk, kh] = max(S_ .* hb);  HR = frq(kh);  prom = pk / median(S_(hb));
    if prom >= 4 && 0.8*HR > 2.5*RR
        fc = min(0.8*HR, max(2.5*RR, 0.5*(2.5*RR + 0.8*HR)));
    else
        HR = NaN;  fc = min(3*RR, 0.4*fps);
    end
    V_lp = fft_lowpass(V, fps, fc, 0.2*fc);
    % ---- breaths on the detrended GN gas volume, reference at the lowest EELV ----
    minsep = round(0.6 * fps / RR);
    vh = mean(V_lp, 2);
    for pass = 1:2
        Y   = calc_difference_data(vh, V_lp, fm);
        hp  = lcurve_hp(Ac, Y);
        rgn = -gn_frames(Ac, Y, hp);
        gV  = Gc.vol' * rgn;
        gd  = gV - movmedian(gV, round(1.5 * fps / RR));
        kmin = find(islocalmin(gd, 'MinSeparation', minsep, 'MinProminence', 0.3*(prc(gd, 95) - prc(gd, 5))));
        [~, lo] = sort(gV(kmin));
        vh  = mean(V_lp(:, kmin(lo(1:min(3, numel(lo))))), 2);
    end
    kmin = kmin(:)';  nb = numel(kmin) - 1;
    kei = zeros(1, nb);
    for b = 1:nb; kk = kmin(b):kmin(b+1);  [~, ip] = max(gV(kk));  kei(b) = kk(ip); end
    TVg  = arrayfun(@(b) gV(kei(b)) - gV(kmin(b)), (1:nb)');
    EELV = gV(kmin(1:nb))';
    tivm = mean(rgn(:, kei) - rgn(:, kmin(1:nb)), 2);
    w    = max(tivm, 0);
    rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);
    fprintf('  RR %.0f /min, HR %.0f /min (prominence %.1f), fc %.2f Hz | %d breaths, insp %.2f s | GN hp %.3g\n', ...
            60*RR, 60*HR, prom, fc, nb, median(kei - kmin(1:nb))*dt, hp);
    % ---- PEEP segmentation of the breath-wise EELV ----
    seg = dp_segment(EELV / median(TVg), numel(PEEPSEQ), 2);       % segment index per breath
    fprintf('  PEEP segmentation (EELV in median-TV units):');
    for s = 1:numel(PEEPSEQ)
        bs = find(seg == s);
        fprintf(' [%s %2d: %d br, %+.1f]', LIMB{s}, PEEPSEQ(s), numel(bs), median(EELV(bs))/median(TVg));
    end
    fprintf('\n');
    segm = arrayfun(@(s) median(EELV(seg == s)), 1:numel(PEEPSEQ));
    [~, itop] = max(segm);
    if itop ~= 6; fprintf('  WARNING: the highest-EELV segment is #%d, expected #6 (PEEP 25) - check the labelling\n', itop); end
    o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
               'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
               'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w);
    conds = {'real'};
    if any(strcmp(RUN_NULL, id)); conds{end+1} = 'null-w'; end %#ok<AGROW>
    res = struct('id', id, 'animal', F{f,2}, 'state', F{f,3}, 'fps', fps, 'RR', RR, 'HR', HR, 'fc', fc, ...
                 'hp', hp, 'kmin', kmin, 'kei', kei, 'TVg', TVg, 'EELV', EELV, 'seg', seg, ...
                 'PEEPSEQ', PEEPSEQ, 'model', MODEL);
    for c = 1:numel(conds)
        if c == 1; Yk = Y; else; Yk = -Ac * (w * (gV / (Gc.vol' * w))) + phase_rand(Y + Ac*rgn, 1); end
        t0 = tic;
        [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
        QG = hk_pendelluft(Gc, m, ant_c, dt);
        Fl = (diff(Gc.vol' * rho) / dt)';
        a  = (Fl' * QG) / (Fl' * Fl);  r = QG - a * Fl;  cc = corrcoef(QG, Fl);
        Bq = 100 * insp_pv(r, kmin, kei, dt) ./ TVg;
        act = (rho + rho0) <= 1e-3 * rho0;
        P = nan(numel(PEEPSEQ), 9);       % [PEEP, nbr, P>A, A>P, lag, TV/TVmed, EELV/TVmed, asym, total]
        for s = 1:numel(PEEPSEQ)
            bs = find(seg == s);  bs = bs(2:end);                % drop the transition breath
            if numel(bs) < 2; P(s,1:2) = [PEEPSEQ(s) numel(bs)]; continue; end
            TT = timing_thirds_b(r, kmin, kei, dt, TVg, bs);
            Lg = (TT(1) + TT(6) - TT(4) - TT(3)) / max(sum(TT([1 3 4 6])), eps);
            mq = median(Bq(bs, :), 1, 'omitnan');
            P(s,:) = [PEEPSEQ(s), numel(bs), mq, Lg, median(TVg(bs))/median(TVg), median(EELV(bs))/median(TVg), ...
                      mq(2)/max(mq(1), eps), sum(mq)];
        end
        R = struct('P', P, 'Bq', Bq, 'a', a, 'cc', cc(1,2), 'QG', QG, 'Fl', Fl, ...
                   'mis', norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), 'misGN', norm(-Ac*rgn - Y, 'fro')/norm(Y, 'fro'), ...
                   'bound', mean(act(:)), 'res', max(info.res), 'iters', info.iters);
        res.(strrep(conds{c}, '-', '_')) = R;
        fprintf('  %-6s | a %+.4f corr %+.2f | misfit %.3f (GN %.3f) bound %.3f | %d it %.0f min\n', conds{c}, a, cc(1,2), ...
                R.mis, R.misGN, R.bound, info.iters, toc(t0)/60);
        fprintf('         %-4s %4s %3s | %6s %6s %6s | %6s | %6s %6s\n', 'limb', 'PEEP', 'nb', 'P>A', 'A>P', 'total', 'lag', 'TV', 'EELV');
        for s = 1:numel(PEEPSEQ)
            fprintf('         %-4s %4d %3d | %6.2f %6.2f %6.2f | %+6.2f | %6.2f %6.2f\n', LIMB{s}, P(s,1), P(s,2), P(s,3), P(s,4), ...
                    P(s,9), P(s,5), P(s,6), P(s,7));
        end
    end
    save(rfile, 'res');
    fprintf('  saved %s (%.0f min)\n', rfile, toc(t_file)/60);
end

% ---------------- summary ----------------
fprintf('\n============ E5 summary: quadrature PV (%% of GN TV per inspiration), median over PEEP levels ============\n');
fprintf('%-9s %-4s %-10s | %6s %6s %6s | %6s %6s | %s\n', 'file', 'anim', 'state', 'P>A', 'A>P', 'total', 'nullT', 'ratio', 'Spearman PEEP vs lag (inc / dec limb pooled), perm p');
for f = 1:size(F, 1)
    rfile = fullfile(outdir, [F{f,1} '.mat']);
    if ~exist(rfile, 'file'); continue; end
    L = load(rfile);  r = L.res;  P = r.real.P;  ok = P(:,2) >= 2;
    mt = median(P(ok, 3:4), 1, 'omitnan');  tot = median(P(ok, 9), 'omitnan');
    if isfield(r, 'null_w'); nt = median(r.null_w.P(ok, 9), 'omitnan'); else; nt = NaN; end
    [rs, pv] = spearman_perm(P(ok,1), P(ok,5), 20000);
    fprintf('%-9s %-4s %-10s | %6.2f %6.2f %6.2f | %6.2f %6.2f | rho %+.2f (p %.3f), n = %d levels\n', F{f,1}, F{f,2}, F{f,3}, ...
            mt, tot, nt, tot / nt, rs, pv, nnz(ok));                 % NaN ratio when no null was run
end
fprintf('H1: ratio >> 1; H2: total(lavaged) > total(normal), total(surfactant) < total(lavaged) in both animals; H3: rho, and inc vs dec limbs.\n');

% ======================================================================
function seg = dp_segment(x, K, minlen)
% optimal piecewise-constant segmentation of x into K contiguous segments
% (least squares, each >= minlen samples) by dynamic programming
x = x(:);  n = numel(x);
cs = [0; cumsum(x)];  cs2 = [0; cumsum(x.^2)];
cost = @(i, j) (cs2(j+1) - cs2(i)) - (cs(j+1) - cs(i))^2 / (j - i + 1);
Cb = inf(K, n);  Bk = zeros(K, n);
for j = minlen:n; Cb(1, j) = cost(1, j); end
for k = 2:K
    for j = k*minlen:n
        best = inf;  arg = 0;
        for i = (k-1)*minlen:j-minlen
            v = Cb(k-1, i) + cost(i+1, j);
            if v < best; best = v; arg = i; end
        end
        Cb(k, j) = best;  Bk(k, j) = arg;
    end
end
seg = zeros(n, 1);  j = n;
for k = K:-1:1
    i = Bk(k, j);  if k == 1; i = 0; end
    seg(i+1:j) = k;  j = i;
end
end

function TT = timing_thirds_b(q, kmin, kei, dt, TVg, bs)
X = nan(numel(bs), 6);
for ii = 1:numel(bs)
    b = bs(ii);  st = kmin(b):kei(b)-1;  n = numel(st);
    if n < 3; continue; end
    u = ((0:n-1) + 0.5) / n;  bin = min(floor(3*u) + 1, 3);
    for h = 1:3
        qs = q(st(bin == h));
        X(ii, h)   = 100 * dt * sum(max(qs, 0))  / TVg(b);
        X(ii, 3+h) = 100 * dt * sum(max(-qs, 0)) / TVg(b);
    end
end
TT = median(X, 1, 'omitnan');
end

function [rho, p] = spearman_perm(x, y, nperm)
% Spearman rank correlation (average ranks for ties) and a two-sided
% permutation p-value
rx = tiedrank_(x(:));  ry = tiedrank_(y(:));
rho = corr_(rx, ry);
rs = RandStream('mt19937ar', 'Seed', 3);  cnt = 0;
for k = 1:nperm
    if abs(corr_(rx, ry(randperm(rs, numel(ry))))) >= abs(rho) - 1e-12; cnt = cnt + 1; end
end
p = (cnt + 1) / (nperm + 1);
end

function r = tiedrank_(x)
[xs, o] = sort(x);  r = zeros(size(x));  n = numel(x);  i = 1;
while i <= n
    j = i;  while j < n && xs(j+1) == xs(i); j = j + 1; end
    r(o(i:j)) = (i + j) / 2;  i = j + 1;
end
end

function c = corr_(a, b)
a = a - mean(a);  b = b - mean(b);  c = (a' * b) / max(norm(a) * norm(b), eps);
end

function q = prc(x, pc)
x = sort(x(isfinite(x)));  n = numel(x);  if n == 0; q = NaN; return; end
r = 1 + (n-1)*pc/100;  lo = floor(r);  hi = ceil(r);
q = x(lo) + (r - lo) * (x(hi) - x(lo));
end

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

