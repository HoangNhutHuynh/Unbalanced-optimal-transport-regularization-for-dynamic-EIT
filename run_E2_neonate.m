%RUN_E2_NEONATE  E2: HK-dEIT (protocol P1) on spontaneously breathing neonate.
%
%   Data: EIDORS contrib "if-neonate-spontaneous" (file P04P-1016.get + .prl,
%   Goe-MF II, 220 frames at 13 fps). 10-day-old neonate, spontaneous
%   breathing, PRONE, head turned to the left. Electrodes: #1 front of the
%   chest, #5 LEFT side, #9 back, #13 right side (numbering towards the
%   subject's LEFT - opposite to the adult E1 data).
%   Data courtesy of I. Frerichs; cite Heinrich S, Schiffmann H, Frerichs A,
%   Klockgether-Radke A, Frerichs I. Intensive Care Med 32:1392-1398, 2006.
%
%   Prone: the ANTERIOR (ventral) lung is gravity-dependent, so the
%   anterior|posterior interface = dependent|non-dependent.
%
%   No ground truth. Steps (no oracle anywhere):
%     1. load + QC; respiratory (RR) and cardiac (HR) rates from the global
%        signal spectrum; cardiac low-pass at fc between 2 RR and 0.8 HR
%     2. neonate_16el_lungs model; electrode orientation from the README
%        convention (rotate so e1 is anterior, flip so e5 is LEFT); sign-
%        aware cardiac map as independent orientation check
%     3. breaths from minima of the GN global gas volume (two passes);
%        reference = mean of the end-expiration frames
%     4. GN-cluster (L-curve hp) and HK P1 (alpha 1e-3, beta 1e-2,
%        delta = 4 L_AP/pi, rho0 = 3 x global TIV, ventilation mode w = TIV)
%     5. per breath (inspiration): HK flux-PV in both directions across the
%        interface, FRIC-vol and region-out on GN images; each as % of the
%        tidal volume of its own image (HK or GN).
%        Repeatability = CV across breaths. GT-free consistency check: HK
%        swing / GN swing.
%
%   ~10 min. Needs EIDORS (and load_real_eit.m, hk_core.m ...).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
fn = fullfile(here, 'Data', 'baby', 'if-neonate-spontaneous (1)', 'P04P-1016.get');

P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;  hp_img = 0.03;
MASK = 'thorax';   % 'lungs' reproduces the lung-mask log in README; 'thorax' (after E2i) writes *_thorax.mat

% ---------------- 1. load, rates, cardiac filter ----------------
D   = load_real_eit(fn);                       % fps from the .prl (13)
fps = D.fps;  dt = 1/fps;  V = D.vv;  T = size(V, 2);
g   = sum(V ./ mean(V, 2) - 1, 1);
[S_, fr] = spectrum(g, fps);
RR  = peak_in(S_, fr, [0.4 1.6]);              % neonate: 25-95 /min
HR  = peak_in(S_, fr, [1.7 3.5]);              % neonate: 100-210 /min
fc  = min(0.8*HR, max(2.2*RR, 0.5*(2*RR + 0.8*HR)));
fprintf('rates: RR %.2f Hz (%.0f /min), HR %.2f Hz (%.0f /min) -> cardiac low-pass fc %.2f Hz\n', ...
        RR, 60*RR, HR, 60*HR, fc);
if 2*RR >= 0.8*HR
    warning('E2: 2nd respiratory harmonic overlaps the cardiac band - filter compromises both');
end
V_lp = fft_lowpass(V, fps, fc, 0.2*fc);
g_lp = sum(V_lp ./ mean(V_lp, 2) - 1, 1);

% ---------------- 2. model + orientation ----------------
fm = mk_library_model('neonate_16el_lungs');
[order, ori] = electrode_orientation(fm, 'left');
fprintf('model: lung areas left %.4f right %.4f | e1 %+.1f deg from anterior | order %s\n', ...
        ori.vol_left, ori.vol_right, ori.e1_offset_deg, mat2str(order));
fm.electrode   = fm.electrode(order);
fm.stimulation = D.stim;
fm = mdl_normalize(fm, 1);
Ne = size(fm.elems, 1);
lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
s0 = ones(Ne,1);  s0(lung_idx) = sig_lung;
J  = calc_jacobian(mk_image(fm, s0));
ce = interp_mesh(fm, 0);  vol = get_elem_volume(fm);

% cardiac map (sign-aware, slab around the electrode plane)
p   = 1 ./ sqrt(sum(J.^2, 1))';  PJt = p .* J';
RMe = PJt / (J*PJt + hp_img^2*eye(size(J,1)));
Vc  = fft_lowpass(V, fps, HR + 0.4, 0.2) - fft_lowpass(V, fps, max(HR - 0.4, fc + 0.1), 0.2);
vh0 = mean(V, 2);
xc  = RMe * calc_difference_data(vh0, vh0 + Vc, fm);
gc  = sum(Vc ./ vh0, 1);
Xc  = xc - mean(xc, 2);  gcc = gc - mean(gc);
cr  = (Xc * gcc') ./ (sqrt(sum(Xc.^2, 2)) * norm(gcc) + eps);
sl  = sign(sum(cr(lung_idx) .* vol(lung_idx)));
nlg = setdiff((1:Ne)', lung_idx);
zel  = mean(cell2mat(arrayfun(@(e) fm.nodes(e.nodes,3), fm.electrode(:), 'UniformOutput', false)));
slab = abs(ce(:,3) - zel) < 0.1 * (max(fm.nodes(:,3)) - min(fm.nodes(:,3)));
sco  = max(-sl * cr, 0) .* std(xc, 0, 2);
cand = nlg(slab(nlg) & cr(nlg) * sl < -0.5);
if ~isempty(cand)
    [~, oc] = sort(sco(cand), 'descend');  hrt = cand(oc(1:max(1, ceil(0.1*numel(cand)))));
    wt = vol(hrt) .* sco(hrt);
    hc = sum(ce(hrt,1:2) .* wt, 1) / sum(wt) - ori.center;
    fprintf('cardiac map: heart at anterior %+.3f, right %+.3f (expect anterior > 0, right < 0)\n', ...
            dot(hc, ori.ant), dot(hc, ori.right));
else
    fprintf('cardiac map: no candidate region found\n');
end

% ---------------- 3. graph, breaths (from GN gas volume), reference ----------------
% Breaths are detected on the GN global gas volume (rho = -dsigma, gas > 0),
% not on the raw voltage sum, so the sign convention of the device does not
% matter. Pass 1: reference = mean of all frames; pass 2: reference = mean
% of the end-expiration frames found in pass 1.
grp = 3 * ones(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
[Gc, sfx] = domain_graph(fm, grp, N_coarse, MASK);
Ac = J * Gc.EL;
minsep = round(0.6 * fps / RR);
vh = mean(V_lp, 2);
for pass = 1:2
    Y   = calc_difference_data(vh, V_lp, fm);
    hp  = lcurve_hp(Ac, Y);
    rgn = -gn_frames(Ac, Y, hp);                % Nc x T, rho = -dsigma
    gV  = Gc.vol' * rgn;
    kmin = find(islocalmin(gV, 'MinSeparation', minsep, 'MinProminence', 0.2*p2p(gV)));
    vh  = mean(V_lp(:, kmin), 2);
end
nb = numel(kmin) - 1;
cs = corrcoef(g_lp, gV);
fprintf('breaths: %d complete (end-expiration frames %s) | GN hp (L-curve) %.2g | corr(voltage sum, GN gas) %+.2f\n', ...
        nb, mat2str(kmin), hp, cs(1,2));
if nb < 3; error('E2: fewer than 3 complete breaths detected'); end

% ---------------- 4. HK P1 ----------------
ant_c = (Gc.cent(:,1:2) - ori.center) * ori.ant' > 0;   % anterior = dependent (prone)
L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/sum(Gc.vol(ant_c)) - ...
             sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/sum(Gc.vol(~ant_c)));
kei = zeros(nb, 1);                             % end-inspiration frame of each breath
tivm = zeros(Gc.N, 1);                          % unclipped TIV map (GN), mean over breaths
for b = 1:nb
    kk = kmin(b):kmin(b+1);  [~, ip] = max(gV(kk));  kei(b) = kk(ip);
    tivm = tivm + (rgn(:, kei(b)) - rgn(:, kk(1))) / nb;
end
w_tidal = max(tivm, 0);
rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, ...
           'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 1000, 'use_eta', false, ...
           'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
[rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Y, Gc, o);
fprintf('HK: anterior fraction of lung volume %.2f, L_AP %.3f, rho0 %.3g, %d it, res %.0e\n', ...
        sum(Gc.vol(ant_c))/sum(Gc.vol), L_ap, rho0, info.iters, max(info.res));
QG = hk_pendelluft(Gc, m, ant_c, dt);           % (T-1) x 1, flux RATE into the anterior region
hV = Gc.vol' * rho;
act = (rho + rho0) <= 1e-3 * rho0;               % positivity bound active (see diag_E1_alpha/pos)
fprintf(['HK vs GN: misfit HK %.3f GN %.3f | positivity bound active: all %.3f, ant %.3f, post %.3f | ' ...
         'GN min gas change / mean TIV: %.2f\n'], ...
        norm(-Ac*rho - Y, 'fro')/norm(Y, 'fro'), norm(-Ac*rgn - Y, 'fro')/norm(Y, 'fro'), ...
        mean(act(:)), mean(mean(act(ant_c,:))), mean(mean(act(~ant_c,:))), ...
        min(rgn(:)) / max(sum(Gc.vol .* tivm) / sum(Gc.vol), eps));

% ---------------- 5. per-breath metrics (inspiration) ----------------
% Each estimator is normalized by the tidal volume of ITS OWN image
% (HK by the HK TV, GN-based ones by the GN TV).
M = nan(nb, 9);   % [TV_GN, TV_HK, HK P->A, HK A->P, FRIC, regout_A, regout_P, swing HK/GN, n frames]
for b = 1:nb
    ki = kmin(b):kei(b);                        % inspiration frames
    st = ki(1):ki(end)-1;                       % transport steps within inspiration
    if numel(st) < 2; continue; end
    TVg = gV(ki(end)) - gV(ki(1));
    TVh = hV(ki(end)) - hV(ki(1));
    pA  = dt * sum(max( QG(st), 0));            % posterior -> anterior (into dependent)
    pP  = dt * sum(max(-QG(st), 0));            % anterior -> posterior
    d   = diff(rgn(:, ki), 1, 2);
    fric = sum(sum(Gc.vol .* max(-d, 0)));
    dA  = Gc.vol(ant_c)' * d(ant_c, :);  dP = Gc.vol(~ant_c)' * d(~ant_c, :);
    swh = p2p((Gc.vol(ant_c)' * rho(ant_c, ki) - Gc.vol(~ant_c)' * rho(~ant_c, ki)) / 2);
    swg = p2p((Gc.vol(ant_c)' * rgn(ant_c, ki) - Gc.vol(~ant_c)' * rgn(~ant_c, ki)) / 2);
    M(b,:) = [TVg, TVh, pA, pP, fric, sum(max(-dA,0)), sum(max(-dP,0)), swh/max(swg,eps), numel(ki)];
end
ok = all(isfinite(M), 2) & M(:,1) > 0 & M(:,2) > 0;
Mp = 100 * [M(ok,3:4) ./ M(ok,2), M(ok,5:7) ./ M(ok,1)];   % % of own TV
names = {'HK post->ant (into dependent)', 'HK ant->post', 'FRIC-vol (GN)', ...
         'region-out anterior (GN)', 'region-out posterior (GN)'};
fprintf('\n============ E2 neonate: per-breath pendelluft, %% of tidal volume (%d breaths) ============\n', nnz(ok));
fprintf('%-32s %8s %8s %8s %8s\n', 'estimator', 'median', 'IQR lo', 'IQR hi', 'CV');
for j = 1:numel(names)
    x = Mp(:, j);
    fprintf('%-32s %8.2f %8.2f %8.2f %8.2f\n', names{j}, median(x), prc(x, 25), prc(x, 75), ...
            std(x) / max(abs(mean(x)), eps));
end
fprintf('inspiration length: median %.0f frames (%.2f s)\n', median(M(ok,9)), median(M(ok,9))*dt);
fprintf('TV ratio HK/GN: median %.2f | swing ratio HK/GN of the A-P half-difference: median %.2f\n', ...
        median(M(ok,2)./M(ok,1)), median(M(ok,8)));
fprintf('global source accounted by the ventilation mode (vent-share): %.2f\n', ...
        p2p([0; cumsum(dt*info.vent_c(:))]) / max(p2p(hV), eps));
save(fullfile(here, ['E2_results' sfx '.mat']), 'M', 'Mp', 'names', 'kmin', 'kei', 'RR', 'HR', 'fc', 'hp', ...
     'rho', 'm', 'rgn', 'QG', 'info', 'order', 'ori', 'P1', 'ant_c');

figure('Name', 'E2 neonate');
t = (0:T-1) * dt;  gsrc = Gc.vol' * info.zeta;
subplot(3,1,1); plot(t, gV, 'k', t, hV, 'b'); hold on; plot(t(kmin), gV(kmin), 'kv', t(kei), gV(kei), 'k^'); hold off;
legend('GN gas volume', 'HK gas volume', 'end-exp', 'end-insp'); xlabel('t [s]');
subplot(3,1,2); plot(t, Gc.vol(ant_c)'*rho(ant_c,:), 'b', t, Gc.vol(~ant_c)'*rho(~ant_c,:), 'r');
legend('anterior (dependent)', 'posterior'); title('HK regional gas content'); xlabel('t [s]');
subplot(3,1,3); plot(t(1:end-1), QG, 'b', t(1:end-1), gsrc, 'k');
legend('HK flux into anterior', 'global source'); title('HK transport vs source (rates)'); xlabel('t [s]');

% ======================================================================
function v = p2p(x); v = max(x) - min(x); end

function q = prc(x, pc)
x = sort(x(:));  n = numel(x);  r = 1 + (n-1)*pc/100;  lo = floor(r);  hi = ceil(r);
q = x(lo) + (r - lo) * (x(hi) - x(lo));
end

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

function [order, ori] = electrode_orientation(fm, towards)
% rotate so that the electrode closest to anterior is e1; flip so that e5
% lies on the subject's side given by 'towards' ('left' for this data set)
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
c_t = wc(1:size(fm.elems,1));
c_l = wc(fm.mat_idx{2});   c_r = wc(fm.mat_idx{3});
rgt = c_r - c_l;            rgt = rgt / norm(rgt);
ant = c_t - (c_l + c_r)/2;  ant = ant - dot(ant, rgt)*rgt;  ant = ant / norm(ant);
ep  = zeros(numel(fm.electrode), 2);
for k = 1:numel(fm.electrode); ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes, 1:2), 1) - c_t; end
a   = atan2(ep*rgt', ep*ant') * 180/pi;
[~, k0] = min(abs(a));  n = size(ep, 1);
order = circshift(1:n, -(k0-1));
side = dot(ep(order(5),:), rgt);                % > 0: right
if (strcmp(towards, 'left') && side > 0) || (strcmp(towards, 'right') && side <= 0)
    order = order([1, n:-1:2]);
end
ori = struct('center', c_t, 'ant', ant, 'right', rgt, 'vol_left', sum(vol(fm.mat_idx{2})), ...
             'vol_right', sum(vol(fm.mat_idx{3})), 'e1_offset_deg', a(order(1)));
end

function [hp, edge] = lcurve_hp(A, Y)
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
edge = k <= 4 || k >= numel(lams) - 3;
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end
