%RUN_E5B_LOCAL  E5b: PEEP-level-LOCAL HK with calibrated nulls (piglets).
%
%   E5 analysed each 180-s manoeuvre in ONE HK run with one ventilation
%   profile w, one reference and one in-phase coefficient a. Post hoc
%   (README, 2026-11-10): the large one-directional quadrature PV (10-29 %TV)
%   was the PEEP-dependent shift of the tidal distribution relative to the
%   manoeuvre-average w (a per PEEP level +0.07..+0.27 at PEEP 0-5, -0.04..
%   -0.13 at PEEP 15-25, |corr(QG,F)| 0.8-1.0); and the circular-shift z of
%   the timing contrast was NOT calibrated (in-phase nulls reached |z| 4.4).
%   Here every selected PEEP level is reconstructed on its own:
%     window  = the breaths of that level from the E5 segmentation (first
%               breath after the step change dropped), frames kmin(first) ..
%               kmin(last+1)
%     local   reference (its end-expiration frames), GN hp (L-curve), w (its
%               TIV map), rho0 = 3 x its mean TIV, in-phase coefficient a
%     real    HK P1 -> quadrature r = QG - aF -> PV per direction, timing
%               contrast C = P>A early + A>P late - A>P early - P>A late (%TV;
%               C > 0: dorsal later), circular-shift z (within the window)
%     nulls   3 in-phase nulls (zero asynchrony, local w, window residual
%               phase-randomized with seeds 1-3) through the SAME pipeline
%   Calibration (summary): the pooled null |C| and null |z| give empirical
%   95th percentiles; a real level counts only if it exceeds them.
%   Files: normal and lavaged, both animals (g08-c003, g08-c017, g10-c005,
%   g10-c019); levels: ZEEP (start, or end if the start has < 2 breaths),
%   PEEP 10 inflation, PEEP 25, PEEP 10 deflation, ZEEP end (L5, added
%   2026-11-10: in both normal files L1 fell back to the END ZEEP, in both
%   lavaged files it is the START ZEEP - L5 gives the lavaged end ZEEP where
%   it has >= 2 breaths; a level whose segment an earlier level of the same
%   file already used is skipped). Missing by design (< 2 breaths after the
%   first is dropped): g08-c003 PEEP10 dec, g10-c005 PEEP10 inc and dec.
%   Needs E5_results\<file>.mat (segmentation). RESUMABLE: E5b_results\.
%   ~64 short HK runs, ~6 h (+ surfactant files ~2-3 h; baselines for done windows: seconds).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
ddir   = fullfile(here, 'Data', 'Pig_torso');
indir  = fullfile(here, 'E5_results');
outdir = fullfile(here, 'E5b_results');  if ~exist(outdir, 'dir'); mkdir(outdir); end
MODEL = 'neonate_16el_lungs';  TOWARDS = 'right';
FILES = {'g08-c003', 'g08-c017', 'g08-c035', 'g10-c005', 'g10-c019', 'g10-c029'};   % + surfactant (2026-11-11)
LEVELS = {'ZEEP', [1 11]; 'PEEP10 inc', 3; 'PEEP25', 6; 'PEEP10 dec', 9; 'ZEEP end', 11};   % E5 segment numbers
NSEED = 3;  NSH = 1000;
MAPS_ONLY = {};  % e.g. {'g08-c003', 11; 'g08-c017', 11; 'g08-c035', 11}:   % non-empty: these (file, E5 segment) only, real HK,
% maps for Fig16 -> maps\E5b_<id>_S<seg>.mat; {} = full E5b. Segment 11 = end ZEEP, present in all three g08 states
if ~isempty(MAPS_ONLY)
    FILES = MAPS_ONLY(:,1)';  LEVELS = {'maps', NaN};
    if ~exist(fullfile(here, 'maps'), 'dir'); mkdir(fullfile(here, 'maps')); end
end
P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_c = 800;  tau0 = 100;  it_hk = 6000;

% ---------------- model (as E5) ----------------
D0 = load_real_eit(fullfile(ddir, [FILES{1} '.get']));
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
rs = RandStream('mt19937ar', 'Seed', 21);

for f = 1:numel(FILES)
    id = FILES{f};
    E5 = load(fullfile(indir, [id '.mat']));  r5 = E5.res;
    D  = load_real_eit(fullfile(ddir, [id '.get']));
    fps = D.fps;  dt = 1/fps;
    V_lp = fft_lowpass(D.vv, fps, r5.fc, 0.2*r5.fc);           % same filter as E5
    kminA = r5.kmin(:)';  keiA = r5.kei(:)';  seg = r5.seg(:);
    used = [];                                                  % E5 segments already analysed for this file
    for L = 1:size(LEVELS, 1)
        rfile = fullfile(outdir, sprintf('%s_L%d.mat', id, L));
        base_only = false;
        if ~isempty(MAPS_ONLY)
            sgu = MAPS_ONLY{f, 2};  bs = find(seg == sgu);  bs = bs(2:end)';
            rfile = fullfile(here, 'maps', sprintf('E5b_%s_S%d.mat', id, sgu));
            if exist(rfile, 'file'); fprintf('%s maps: done (skipped)\n', id); continue; end
            if numel(bs) < 2; fprintf('%s maps: < 2 breaths in segment %d - skipped\n', id, sgu); continue; end
        elseif exist(rfile, 'file')
            S = load(rfile);  used(end+1) = S.R.segment; %#ok<AGROW>
            if isfield(S.R.real, 'base'); fprintf('%s %s: done (skipped)\n', id, LEVELS{L,1}); continue; end
            R0 = S.R;  base_only = true;  sgu = R0.segment;       % HK done before the baselines existed: add them only
            bs = find(seg == sgu);  bs = bs(2:end)';
        else
            bs = [];
            for sgi = LEVELS{L,2}
                cand = find(seg == sgi);  cand = cand(2:end)';
                if numel(cand) >= 2; bs = cand; sgu = sgi; break; end
            end
            if isempty(bs); fprintf('%s %s: < 2 breaths - skipped\n', id, LEVELS{L,1}); continue; end
            if ismember(sgu, used); fprintf('%s %s: segment %d already analysed - skipped\n', id, LEVELS{L,1}, sgu); continue; end
            used(end+1) = sgu; %#ok<AGROW>
        end
        w0 = kminA(bs(1));  w1 = kminA(bs(end) + 1);
        win = w0:w1;  T = numel(win);
        kmin = kminA(bs(1):bs(end)+1) - w0 + 1;  kei = keiA(bs) - w0 + 1;  nb = numel(bs);
        % ---- local reference, GN, w, rho0 ----
        vh  = mean(V_lp(:, kminA(bs(1):bs(end)+1)), 2);
        Y   = calc_difference_data(vh, V_lp(:, win), fm);
        hp  = lcurve_hp(Ac, Y);
        rgn = -gn_frames(Ac, Y, hp);
        gV  = Gc.vol' * rgn;
        TVg = arrayfun(@(b) gV(kei(b)) - gV(kmin(b)), (1:nb)');
        tivm = mean(rgn(:, kei) - rgn(:, kmin(1:nb)), 2);
        w    = max(tivm, 0);
        rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);
        E    = Y + Ac * rgn;
        o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
                   'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
                   'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w);
        fprintf('\n%s (%s) %s [E5 segment %d]: %d breaths, %d frames, GN hp %.3g\n', id, r5.state, LEVELS{L,1}, sgu, nb, T, hp);
        R = struct('id', id, 'state', r5.state, 'level', LEVELS{L,1}, 'segment', sgu, 'nb', nb, 'T', T, 'hp', hp);
        if base_only; R = R0; end
        for c = 0:NSEED
            t0 = tic;
            if c == 0; Yk = Y; nm = 'real'; else; Yk = -Ac * (w * (gV / (Gc.vol' * w))) + phase_rand(E, c); nm = sprintf('null%d', c); end
            [Bb, lagb] = pend_estimators(-gn_frames(Ac, Yk, hp), Gc.vol, ant_c, kmin, kei, dt);   % image baselines
            base = struct('B', median(Bb, 1, 'omitnan'), 'lag', lagb);
            if base_only
                R.(nm).base = base;
                fprintf('  %-5s | baselines opp %.2f quadGN %.2f FRIC %.2f regout %.2f Sang %.2f %%TV | lag %+.0f ms\n', nm, base.B, lagb);
                continue
            end
            [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
            QG = hk_pendelluft(Gc, m, ant_c, dt);
            F  = (diff(Gc.vol' * rho) / dt)';
            a  = (F' * QG) / (F' * F);  q = QG - a * F;  cc = corrcoef(QG, F);
            Bq = 100 * insp_pv(q, kmin, kei, dt) ./ TVg;
            Cv = contrast(q, kmin, kei, dt, TVg);
            P  = median(diff(kmin));
            Cs = zeros(NSH, 1);
            for k = 1:NSH
                Cs(k) = contrast(circshift(q, mod(round((randi(rs, [0 2]) + 0.25 + 0.5*rand(rs)) * P), numel(q))), kmin, kei, dt, TVg);
            end
            z  = (Cv - mean(Cs)) / max(std(Cs), eps);
            act = (rho + rho0) <= 1e-3 * rho0;
            out = struct('a', a, 'cc', cc(1,2), 'PV', median(Bq, 1, 'omitnan'), 'C', Cv, 'z', z, ...
                         'mis', norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), 'bound', mean(act(:)), 'iters', info.iters, 'base', base);
            R.(nm) = out;
            if ~isempty(MAPS_ONLY)
                MP = pend_maps(Gc, rgn, rho, m, kmin, kei, ant_c, dt);
                MP.label = sprintf('Piglet %s (%s), E5 segment %d (end ZEEP)', id, r5.state, sgu);
                save(rfile, 'MP', 'fm', 'Gc');
                fprintf('  maps saved %s (HK quadrature %.2f %%TV)\n', rfile, median(MP.PVq));
                break
            end
            fprintf('  %-5s | a %+.3f corr %+.2f | PV P>A %5.2f A>P %5.2f %%TV | C %+6.2f %%TV  z %+5.1f | misfit %.3f bound %.3f | %d it %.1f min\n', ...
                    nm, a, cc(1,2), out.PV, Cv, z, out.mis, out.bound, info.iters, toc(t0)/60);
        end
        if ~isempty(MAPS_ONLY); continue; end
        save(rfile, 'R');
    end
end
if ~isempty(MAPS_ONLY); return; end

% ---------------- calibrated summary ----------------
fr = dir(fullfile(outdir, '*.mat'));
Rall = cell(numel(fr), 1);
for k = 1:numel(fr); S = load(fullfile(fr(k).folder, fr(k).name)); Rall{k} = S.R; end
Rall = [Rall{:}];
nullC = [];  nullZ = [];  nullPV = [];
for k = 1:numel(Rall)
    for c = 1:NSEED
        nm = sprintf('null%d', c);
        if isfield(Rall(k), nm) && ~isempty(Rall(k).(nm))
            nullC(end+1) = Rall(k).(nm).C;  nullZ(end+1) = Rall(k).(nm).z;  nullPV(end+1) = sum(Rall(k).(nm).PV); %#ok<AGROW>
        end
    end
end
tC = prc(abs(nullC), 95);  tZ = prc(abs(nullZ), 95);  tP = prc(nullPV, 95);
fprintf('\n============ E5b calibration from %d null runs: p95 |C| %.2f %%TV, p95 |z| %.2f, p95 total PV %.2f %%TV ============\n', ...
        numel(nullC), tC, tZ, tP);
fprintf('%-9s %-8s %-11s %3s | %6s %5s | %6s %6s %6s | %7s %5s | %7s %s\n', 'file', 'state', 'level', 'nb', 'a', 'corr', 'P>A', 'A>P', 'total', 'C', 'z', 'null C', 'flags');
for k = 1:numel(Rall)
    x = Rall(k);  rr = x.real;
    nc = arrayfun(@(c) x.(sprintf('null%d', c)).C, 1:NSEED);
    flg = '';
    if abs(rr.C) > tC; flg = [flg ' |C|>p95']; end %#ok<AGROW>
    if abs(rr.z) > tZ; flg = [flg ' |z|>p95']; end %#ok<AGROW>
    if sum(rr.PV) > tP; flg = [flg ' PV>p95']; end %#ok<AGROW>
    fprintf('%-9s %-8s %-11s %3d | %+6.3f %+5.2f | %6.2f %6.2f %6.2f | %+7.2f %+5.1f | %s %s\n', x.id, x.state, x.level, x.nb, ...
            rr.a, rr.cc, rr.PV, sum(rr.PV), rr.C, rr.z, mat2str(nc, 2), flg);
end
fprintf('C > 0: dorsal (dependent) region later.  A level is interpretable only if it exceeds the calibrated null thresholds.\n');

% ---------------- image baselines vs the same nulls ----------------
nmB = {'opp', 'quadGN', 'FRIC', 'regout', 'Sang'};
NBm = cell2mat(arrayfun(@(x) cell2mat(arrayfun(@(c) x.(sprintf('null%d', c)).base.B, (1:NSEED)', 'UniformOutput', false)), Rall(:), 'UniformOutput', false));
tB  = arrayfun(@(j) prc(NBm(:, j), 95), 1:numel(nmB));
fprintf('\n============ baselines on GN images (%%TV), * = above pooled null p95 [%s] ============\n', num2str(tB, '%.2f '));
fprintf('%-9s %-10s %-11s | %8s %8s %8s %8s %8s | %7s | HK quad total (p95 %.2f)\n', 'file', 'state', 'level', nmB{:}, 'lag ms', tP);
for k = 1:numel(Rall)
    x = Rall(k);  b = x.real.base.B;  st = repmat(' ', 1, numel(b));  st(b > tB) = '*';
    fprintf('%-9s %-10s %-11s |', x.id, x.state, x.level);  fprintf(' %7.2f%c', [b; double(st)]);
    mk = ' ';  if sum(x.real.PV) > tP; mk = '*'; end
    fprintf(' | %+7.0f | %.2f%c\n', x.real.base.lag, sum(x.real.PV), mk);
end

% ======================================================================
function C = contrast(q, kmin, kei, dt, TVg)
X = nan(numel(kei), 6);
for b = 1:numel(kei)
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
C  = TT(1) + TT(6) - TT(4) - TT(3);
end

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

