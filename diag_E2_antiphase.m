%DIAG_E2_ANTIPHASE  Where is the component that loses "gas" during inspiration?
%
%   E2b: on the real neonate data (not on in-phase null surrogates) the GN
%   image has nodes whose gas content DROPS during inspiration by a total of
%   ~2.6 TV (FRIC-net 259 %TV vs 27-49 %TV on the nulls), HK cannot fit it
%   (misfit 0.318 vs 0.138 on the nulls) and the positivity bound becomes
%   active (5 % vs 0 %). Region-out is 0 in both anterior and posterior, so
%   the loss is WITHIN the A|P regions.
%
%   Element-level GN over the WHOLE domain (lungs and non-lung tissue), on
%   the real data and on an in-phase null surrogate (ringing reference).
%   Per element: inspiratory change per breath, mean over breaths (TIV_e)
%   and sign consistency. Anti-phase set = elements in the electrode slab
%   with TIV_e < 0 in >= 75 % of the breaths. Reported:
%     - integrated loss of the set (% of the slab TV) split into
%       left lung / right lung / non-lung tissue
%     - inside the lungs: anterior vs posterior, peripheral (near the chest
%       wall) vs central (mediastinal)
%     - loss-weighted centroid (anterior, right coordinates; E2 cardiac map
%       put the heart at anterior +0.518, right -0.218)
%     - most negative TIV_e in units of the mean lung TIV per unit volume
%     - lag of the anti-phase content relative to the global signal
%   Figures: TIV_e maps (real, null) at the electrode plane, time courses.
%
%   ~3 min, GN only. Needs EIDORS.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
fn = fullfile(here, 'Data', 'baby', 'if-neonate-spontaneous (1)', 'P04P-1016.get');
sig_lung = 0.4;  N_coarse = 800;  cons_min = 0.75;  periph = 0.15;

% ---------------- data, filter, model (as run_E2_neonate) ----------------
D   = load_real_eit(fn);
fps = D.fps;  V = D.vv;  T = size(V, 2);
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
kei = zeros(nb, 1);  tivm = zeros(Gc.N, 1);
for b = 1:nb
    kk = kmin(b):kmin(b+1);  [~, ip] = max(gV(kk));  kei(b) = kk(ip);
    tivm = tivm + (rgn(:, kei(b)) - rgn(:, kk(1))) / nb;
end
w_tidal = max(tivm, 0);

% in-phase null surrogate (as null-w r1 in run_E2b_floor)
Ynull = -Ac * (w_tidal * (gV / (Gc.vol' * w_tidal))) + phase_rand(Y + Ac*rgn, 1);

% ---------------- geometry ----------------
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
[ant, rgt, c_t] = axes_dir(fm);
zel  = mean(cell2mat(arrayfun(@(e) fm.nodes(e.nodes,3), fm.electrode(:), 'UniformOutput', false)));
H    = max(fm.nodes(:,3)) - min(fm.nodes(:,3));
slab = abs(ce(:,3) - zel) < 0.25 * H;
isL  = false(Ne,1);  isL(fm.mat_idx{2}) = true;
isR  = false(Ne,1);  isR(fm.mat_idx{3}) = true;
isLung = isL | isR;
bn   = unique(fm.boundary(:));
bn   = bn(abs(fm.nodes(bn,3) - zel) < 0.25 * H);
bxy  = fm.nodes(bn, 1:2);
Req  = mean(sqrt(sum((bxy - c_t).^2, 2)));
dwall = zeros(Ne, 1);
for k = 1:Ne; dwall(k) = sqrt(min(sum((bxy - ce(k,1:2)).^2, 2))); end
isPer = dwall < periph * Req;
pa = (ce(:,1:2) - c_t) * ant';   pr = (ce(:,1:2) - c_t) * rgt';
fprintf('slab: %d elements | lung %.0f %% of slab volume | Req %.3f\n', nnz(slab), ...
        100*sum(vol(slab & isLung))/sum(vol(slab)), Req);

% ---------------- element-level GN, whole domain ----------------
hpe = lcurve_hp(J, Y);
fprintf('element GN hp (L-curve) %.3g | cluster GN hp %.3g\n', hpe, hp);
lab = {'real', 'null-w'};  YY = {Y, Ynull};  TIVs = cell(1,2);  Acon = cell(1,2);
for c = 1:2
    x  = -gn_frames(J, YY{c}, hpe);            % gas = -dsigma, Ne x T
    dI = x(:, kei) - x(:, kmin(1:nb));         % Ne x nb inspiratory change
    TIVe = mean(dI, 2);  cons = mean(dI < 0, 2);
    TIVs{c} = TIVe;
    TV  = sum(vol(slab) .* TIVe(slab));
    tl  = sum(vol(slab & isLung) .* TIVe(slab & isLung)) / sum(vol(slab & isLung));
    neg = slab & TIVe < 0 & cons >= cons_min;
    lw  = -vol .* TIVe .* neg;                 % loss weight (>= 0)
    Ls  = @(sel) 100 * sum(lw(sel)) / TV;
    fprintf('\n---- %s ----\n', lab{c});
    fprintf('slab TV %.4g | mean lung TIV per volume %.4g | anti-phase set: %.1f %% of slab volume\n', ...
            TV, tl, 100*sum(vol(neg))/sum(vol(slab)));
    fprintf('integrated loss %% of TV: total %.1f | left lung %.1f | right lung %.1f | non-lung %.1f\n', ...
            Ls(neg), Ls(neg & isL), Ls(neg & isR), Ls(neg & ~isLung));
    fprintf('  inside lungs: anterior %.1f posterior %.1f | peripheral (< %.0f %% Req from wall) %.1f central %.1f\n', ...
            Ls(neg & isLung & pa > 0), Ls(neg & isLung & pa <= 0), 100*periph, ...
            Ls(neg & isLung & isPer), Ls(neg & isLung & ~isPer));
    fprintf('  non-lung: peripheral %.1f central %.1f\n', Ls(neg & ~isLung & isPer), Ls(neg & ~isLung & ~isPer));
    if any(neg)
        fprintf('  loss-weighted centroid: anterior %+.3f right %+.3f | most negative TIV_e / mean lung TIV: %.1f\n', ...
                sum(lw.*pa)/sum(lw), sum(lw.*pr)/sum(lw), min(TIVe(slab)) / tl);
        a  = vol(neg)' * x(neg, :);  gl = vol(slab)' * x(slab, :);
        [lg, cc] = best_lag(a, gl, 5);
        fprintf('  anti-phase content vs global: corr %+.2f at lag %+d frames (%.2f s; > 0: content lags)\n', ...
                cc, lg, lg/fps);
        Acon{c} = a;
    end
end
save(fullfile(here, 'diag_E2_antiphase.mat'), 'TIVs', 'Acon', 'kmin', 'kei', 'hpe', 'zel');

figure('Name', 'E2 anti-phase');
for c = 1:2
    subplot(2,2,c);
    img = mk_image(fm, TIVs{c});  img.calc_colours.ref_level = 0;
    show_slices(img, [inf inf zel]);  title(['TIV_e ' lab{c} ' (electrode plane)']);
end
subplot(2,1,2);  t = (0:T-1)/fps;
if ~isempty(Acon{1})
    yyaxis left;  plot(t, Acon{1});  ylabel('anti-phase set content');
end
yyaxis right; plot(t, Gc.vol' * rgn);  ylabel('global gas (cluster GN)');  xlabel('t [s]');

% ======================================================================
function [lg, cc] = best_lag(a, b, maxlag)
a = a(:) - mean(a);  b = b(:) - mean(b);  T = numel(a);
best = 0;  lg = 0;  cc = 0;
for L = -maxlag:maxlag
    if L >= 0; i1 = 1+L:T; i2 = 1:T-L; else; i1 = 1:T+L; i2 = 1-L:T; end
    r = (a(i1)' * b(i2)) / (norm(a(i1)) * norm(b(i2)) + eps);
    if abs(r) > best; best = abs(r); lg = L; cc = r; end
end
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
