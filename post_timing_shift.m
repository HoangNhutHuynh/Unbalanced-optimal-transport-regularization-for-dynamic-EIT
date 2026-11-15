%POST_TIMING_SHIFT  Timing contrast of the quadrature flux with a circular-
%   shift null - post hoc on saved E3 (paediatric ARDS) and E5 (piglet)
%   results. No reconstruction is re-run.
%
%   Why: the lag index L = (P>A early + A>P late - A>P early - P>A late) / sum
%   saturates at +-1 even for noise (E3 v2, patient 1: the circular-shift null
%   of L spans most of [-1, 1]; one in-phase null even gave L = +0.83). The
%   timing CONTRAST C = P>A early + A>P late - A>P early - P>A late (in %TV,
%   not normalized) does not saturate.
%   Sign rule (E2g v2): C > 0 <=> the POSTERIOR region fills later.
%   Null: the quadrature flux r(t) is circularly shifted by a random non-
%   integer number of breath periods (fractional part 0.25-0.75), which keeps
%   its amplitude and spectrum but destroys its locking to the breath cycle;
%   C is recomputed on the same breaths. z = (C - mean_null) / sd_null and a
%   two-sided p = P(|C_null| >= |C|).
%   Patient 1 (E3 v2), first post hoc check: C positive at PEEP >= 20
%   (z +2.2 .. +3.2), negative at PEEP 14 on both limbs (z -2.2, -2.3);
%   Spearman PEEP vs z rho 0.91.
%
%   Output: tables in the Command Window + post_timing_shift.mat.  ~1 min.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
NSH = 2000;  rs = RandStream('mt19937ar', 'Seed', 11);
OUT = struct('set', {}, 'id', {}, 'state', {}, 'PEEP', {}, 'limb', {}, 'nb', {}, 'C', {}, 'z', {}, 'p', {}, 'Cnull', {}, 'znull', {});

% ---------------- E3: one PEEP level per file ----------------
f3 = dir(fullfile(here, 'E3_results', '*_thorax_v2.mat'));
for k = 1:numel(f3)
    L = load(fullfile(f3(k).folder, f3(k).name));  r = L.res;
    bs = 1:numel(r.kei);
    [C, z, p] = contrast_null(r.real, r.kmin, r.kei, 1/r.fps, r.TVg, bs, NSH, rs);
    [Cn, zn] = deal(NaN);
    if isfield(r, 'null_w'); [Cn, zn] = contrast_null(r.null_w, r.kmin, r.kei, 1/r.fps, r.TVg, bs, NSH, rs); end
    OUT(end+1) = struct('set', 'E3', 'id', r.id, 'state', r.maneuver, 'PEEP', r.PEEP, 'limb', r.maneuver, ...
                        'nb', numel(bs), 'C', C, 'z', z, 'p', p, 'Cnull', Cn, 'znull', zn); %#ok<SAGROW>
end

% ---------------- E5: one row per PEEP level per manoeuvre ----------------
f5 = dir(fullfile(here, 'E5_results', '*.mat'));
LIMB = {'0', 'inc', 'inc', 'inc', 'inc', 'top', 'dec', 'dec', 'dec', 'dec', 'end'};
for k = 1:numel(f5)
    L = load(fullfile(f5(k).folder, f5(k).name));  r = L.res;
    for s = 1:numel(r.PEEPSEQ)
        bs = find(r.seg == s);  bs = bs(2:end)';
        if numel(bs) < 2; continue; end
        [C, z, p] = contrast_null(r.real, r.kmin, r.kei, 1/r.fps, r.TVg, bs, NSH, rs);
        [Cn, zn] = deal(NaN);
        if isfield(r, 'null_w'); [Cn, zn] = contrast_null(r.null_w, r.kmin, r.kei, 1/r.fps, r.TVg, bs, NSH, rs); end
        OUT(end+1) = struct('set', 'E5', 'id', r.id, 'state', r.state, 'PEEP', r.PEEPSEQ(s), 'limb', LIMB{s}, ...
                            'nb', numel(bs), 'C', C, 'z', z, 'p', p, 'Cnull', Cn, 'znull', zn); %#ok<SAGROW>
    end
end
save(fullfile(here, 'post_timing_shift.mat'), 'OUT');

% ---------------- tables ----------------
fprintf('%-4s %-9s %-10s %4s %-4s %3s | %7s %6s %6s | %7s %6s\n', 'set', 'id', 'state', 'PEEP', 'limb', 'nb', 'C %TV', 'z', 'p', 'C null', 'z null');
for i = 1:numel(OUT)
    o = OUT(i);
    fprintf('%-4s %-9s %-10s %4d %-4s %3d | %+7.2f %+6.1f %6.3f | %+7.2f %+6.1f\n', o.set, o.id, o.state, o.PEEP, o.limb, ...
            o.nb, o.C, o.z, o.p, o.Cnull, o.znull);
end
ids = unique({OUT.id}, 'stable');
grp = unique(cellfun(@(s, i) [s ':' i], {OUT.set}, {OUT.id}, 'UniformOutput', false), 'stable');
fprintf('\nper recording set (E3: patient 1 pooled; E5: per manoeuvre): Spearman PEEP vs z (permutation)\n');
e3 = strcmp({OUT.set}, 'E3');
if nnz(e3) >= 4
    [rh, pv] = spearman_perm([OUT(e3).PEEP], [OUT(e3).z], 20000);
    fprintf('  E3 patient 1: rho %+.2f (p %.4f), n = %d, |z| >= 2 in %d\n', rh, pv, nnz(e3), nnz(abs([OUT(e3).z]) >= 2));
end
for i = 1:numel(ids)
    sel = strcmp({OUT.set}, 'E5') & strcmp({OUT.id}, ids{i});
    if nnz(sel) < 4; continue; end
    [rh, pv] = spearman_perm([OUT(sel).PEEP], [OUT(sel).z], 20000);
    fprintf('  E5 %-9s (%s): rho %+.2f (p %.4f), n = %d levels, |z| >= 2 in %d\n', ids{i}, OUT(find(sel, 1)).state, ...
            rh, pv, nnz(sel), nnz(abs([OUT(sel).z]) >= 2));
end
clear grp

% ======================================================================
function [C, z, p] = contrast_null(R, kmin, kei, dt, TVg, bs, nsh, rs)
r  = R.QG(:) - R.a * R.Fl(:);
kmin = kmin(:)';  kei = kei(:)';
C  = contrast(r, kmin, kei, dt, TVg, bs);
P  = median(diff(kmin));  T = numel(r);
Cs = zeros(nsh, 1);
for k = 1:nsh
    s = mod(round((randi(rs, [0 5]) + 0.25 + 0.5*rand(rs)) * P), T);
    Cs(k) = contrast(circshift(r, s), kmin, kei, dt, TVg, bs);
end
z = (C - mean(Cs)) / max(std(Cs), eps);
p = mean(abs(Cs) >= abs(C) - 1e-12);
end

function C = contrast(q, kmin, kei, dt, TVg, bs)
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
C  = TT(1) + TT(6) - TT(4) - TT(3);
end

function [rho, p] = spearman_perm(x, y, nperm)
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
