%DIAG_E3_QUALITY  Why does patient 7 fail, and are patient 1's filters right?
%
%   E3 v1: patient 7 (13.9 y, 76 kg, ECMO; adult_male model): GN L-curve hp
%   at the lower edge (1e-4), GN misfit 0.55-0.67, HK misfit 0.77-0.90 and
%   HK tidal volume ~0 in every file, while the in-phase null fits (0.02) -
%   the measured data are not representable by the model. Patient 1: the
%   cardiac peak was detected at the band edge (HR 54-62 /min in a 5.9-y
%   child), which set the low-pass to 0.67-0.72 Hz, BELOW the second
%   respiratory harmonic (RR 26 /min -> 0.87 Hz): the breath waveform and
%   thus the quadrature timing are distorted.
%   For every recording (last 20 s, as E3 v1):
%     - respiratory peak; cardiac peak and its PROMINENCE (peak / median of
%       the band), i.e. whether a heart rate is detectable at all
%     - ventilation-to-noise ratio: median over channels of the p2p of the
%       0-3RR band vs the std of the residual above 3RR
%     - reciprocity error of the mean frame (drive pair <-> measurement pair
%       swap; large values = electrode contact / hardware problem)
%     - model consistency: element GN on the whole domain at a FIXED relative
%       regularization (lambda = 1e-2 mean eigenvalue) for three model
%       variants per patient - library model with numbering towards the
%       patient's left (as documented) / towards the right / the other
%       library thorax - misfit and explained variance; plus the L-curve hp
%       and whether it sits at the edge of the scanned range.
%
%   ~5 min, GN only.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
root = fullfile(here, 'Data', 'child');
p1 = 'cg_data_2012_p1/DATA/STUDYNAME/SUBJECT_1/YYYYMMDD/Eit/Viasys/';
p7 = 'cg_data_2012_p7/DATA/STUDYNAME/SUBJECT_7/YYYYMMDD/Eit/Viasys/';
F = {'1001_b',1,p1,[]; '1001_c1',1,p1,[]; '1001_c2',1,p1,[]; '1001_c3',1,p1,[]; '1001_c4',1,p1,[]; ...
     '1001_d1',1,p1,[1 765]; '1001_d2',1,p1,[]; '1001_d3',1,p1,[368 780]; '1001_d4',1,p1,[411 780]; ...
     '1007_b2',7,p7,[]; '1007_c1',7,p7,[]; '1007_c2',7,p7,[]; '1007_c3',7,p7,[]; '1007_c4',7,p7,[]; ...
     '1007_d1',7,p7,[]; '1007_d2',7,p7,[]};
VAR = struct('subj', {1, 1, 1, 7, 7, 7}, ...
             'model', {'neonate_16el_lungs', 'neonate_16el_lungs', 'adult_male_16el_lungs', ...
                       'adult_male_16el_lungs', 'adult_male_16el_lungs', 'neonate_16el_lungs'}, ...
             'towards', {'left', 'right', 'left', 'left', 'right', 'left'});
WIN_S = 20;  fps0 = 13;

% ---------------- per-file signal quality ----------------
fprintf('%-8s | %5s %5s %6s | %7s | %8s | %s\n', 'file', 'RR/m', 'HR/m', 'prom', 'VNR', 'recip', 'note');
Ys = cell(size(F,1), 1);  stim = [];
for f = 1:size(F,1)
    D = load_real_eit(fullfile(root, F{f,3}, [F{f,1} '.get']), struct('fps', fps0));
    if isempty(stim); stim = D.stim; end
    fps = D.fps;  rg = F{f,4};  if isempty(rg); rg = [1 size(D.vv,2)]; end
    nW = min(round(WIN_S*fps), rg(2)-rg(1)+1);  fr = rg(2)-nW+1:rg(2);
    V = D.vv(:, fr);
    g = sum(V ./ mean(V, 2) - 1, 1);
    [S, fq] = spectrum(g, fps);
    in = fq >= 0.15 & fq <= 1.0;  [~, k] = max(S .* in);  RR = fq(k);
    hb = fq >= max(1.6*RR, 0.75) & fq <= min(3.5, 0.45*fps);
    [pk, k2] = max(S .* hb);  HR = fq(k2);  prom = pk / median(S(hb));
    Vl = fft_lowpass(V, fps, 3*RR, 0.3*RR);
    vnr = median(p2p(Vl') ./ std(V - Vl, 0, 2)');
    rec = reciprocity_error(mean(V, 2), D.stim);
    note = '';
    if prom < 4; note = 'no clear cardiac peak'; end
    fprintf('%-8s | %5.0f %5.0f %6.1f | %7.1f | %8.3f | %s\n', F{f,1}, 60*RR, 60*HR, prom, vnr, rec, note);
    Ys{f} = (V - mean(V, 2)) ./ mean(V, 2);                 % normalized difference data (as mdl_normalize 1)
end

% ---------------- model consistency ----------------
fprintf('\n%-24s %-6s | %s\n', 'model variant', 'subj', 'per file: misfit at fixed lambda (L-curve hp, EDGE flag)');
for v = 1:numel(VAR)
    fm = mk_library_model(VAR(v).model);
    fm.electrode = fm.electrode(electrode_order(fm, VAR(v).towards));
    fm.stimulation = stim;  fm = mdl_normalize(fm, 1);
    Ne = size(fm.elems, 1);  s0 = ones(Ne,1);  s0([fm.mat_idx{2}(:); fm.mat_idx{3}(:)]) = 0.4;
    J = calc_jacobian(mk_image(fm, s0));
    p = 1 ./ sqrt(sum(J.^2, 1))';  SS = J * (p .* J');  SS = (SS + SS')/2;
    [U, Dg] = eig(SS);  s = max(diag(Dg), 0);  lam = 1e-2 * mean(s);
    line = sprintf('%-24s %-6d |', sprintf('%s/%s', strrep(VAR(v).model, '_16el_lungs', ''), VAR(v).towards), VAR(v).subj);
    for f = find([F{:,2}] == VAR(v).subj)
        Y = Ys{f};
        Z2 = sum((U' * Y).^2, 2);
        mis = sqrt(sum((lam ./ (s + lam)).^2 .* Z2) / sum(Z2));
        [hp, edge] = lcurve_hp_edge(s, Z2);
        line = [line sprintf(' %s %.3f (%.1e%s)', F{f,1}(6:end), mis, hp, ternary(edge, ' EDGE', ''))]; %#ok<AGROW>
    end
    fprintf('%s\n', line);
end
fprintf(['\nReading: a model variant that explains the data clearly better (lower misfit at the same lambda)\n' ...
         'indicates the right shape / numbering; reciprocity error >> that of patient 1 or VNR << 10 points to\n' ...
         'the data (contact, ECMO interference) rather than the model.\n']);

% ======================================================================
function e = reciprocity_error(v, stim)
% relative reciprocity error of one frame: swap drive and measurement pairs
P = [];  
for k = 1:numel(stim)
    d = full(stim(k).stim_pattern(:))';  M = full(stim(k).meas_pattern);
    for j = 1:size(M,1); P = [P; d, M(j,:)]; end %#ok<AGROW>
end
n = size(P,2) / 2;  Dd = sign(P(:,1:n));  Mm = sign(P(:,n+1:end));
num = 0;  den = 0;
for i = 1:size(P,1)
    j = find(all(Dd == Mm(i,:), 2) & all(Mm == Dd(i,:), 2), 1);
    sg = 1;
    if isempty(j)
        j = find(all(Dd == -Mm(i,:), 2) & all(Mm == -Dd(i,:), 2), 1);  sg = 1;
        if isempty(j)
            j = find(all(Dd == -Mm(i,:), 2) & all(Mm == Dd(i,:), 2), 1);  sg = -1;
        end
        if isempty(j)
            j = find(all(Dd == Mm(i,:), 2) & all(Mm == -Dd(i,:), 2), 1);  sg = -1;
        end
    end
    if isempty(j); continue; end
    num = num + (v(i) - sg*v(j))^2;  den = den + v(i)^2;
end
e = sqrt(num / max(den, eps));
end

function [hp, edge] = lcurve_hp_edge(s, Z2)
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
[~, k] = max(kap);  hp = sqrt(lams(k));  edge = k <= 6 || k >= numel(lams) - 5;
end

function s = ternary(c, a, b); if c; s = a; else; s = b; end; end

function v = p2p(x); v = max(x) - min(x); end

function [S, f] = spectrum(g, fps)
g = g(:)';  T = numel(g);
g = g - polyval(polyfit(1:T, g, 1), 1:T);
w = 0.5 - 0.5*cos(2*pi*(0:T-1)/(T-1));
nf = 8*2^nextpow2(T);
S = abs(fft(g .* w, nf));  f = (0:nf-1) / nf * fps;
keep = f <= fps/2;  S = S(keep);  f = f(keep);
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

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end

