%DIAG_E1_POS  Is the HK swing loss on the paradoxical record (1pl 003) due to
%   the positivity bound rho + rho0 >= 0?
%
%   diag_E1_alpha: in 003, kappa 3 -> 30 halves the bound-active fraction
%   (0.64 -> 0.036), doubles the swing ratio (0.24 -> 0.49) and lowers the
%   misfit (0.52 -> 0.33; GN 0.225); 100x more action weight (alpha, which
%   multiplies the HK action) changes the swing by < 10%.
%   NOTE: alpha weights the ACTION, so shrinkage (a) would show as the ratio
%   rising when alpha DEcreases - tested here with alpha = 1e-4.
%
%   Grid (1-plane, anterior|posterior, beta 1e-3, max_iter 8000):
%     003: kappa in {30, 100, 1000} x alpha in {1e-4, 1e-3}
%     001: kappa 1000, alpha 1e-3 (control)
%   Per run:
%     swing HK/GN, swing clipGN/GN with clipGN = max(rho_GN, -rho0) (what the
%     bound alone allows), misfit HK/GN, bound-active fraction overall and
%     in the anterior / posterior region, T-share, flux-PV, res, iterations.
%   Reading:
%     HK ~ clipGN for every kappa        -> loss is the bound only
%     ratio -> ~1 and misfit -> GN at kappa 1000 -> GN anti-phase signal needs
%        a gas loss larger than kappa x mean TIV per unit volume, i.e. it is
%        not (only) gas - e.g. chest-wall / boundary motion
%     ratio rises at alpha 1e-4             -> action shrinkage contributes
%
%   ~20 min. Needs E1_prep.mat and EIDORS.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
S = load(fullfile(here, 'E1_prep.mat'));
recs = S.recs_save;  fps = S.fps;

beta = 1e-3;  win_s = 20;  ds = 2;  dt = ds / fps;
dm = 4;  sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 8000;
% stage 1 (done, see README; saved as diag_E1_pos.mat): positivity, kappa x alpha at beta 1e-3
% runs = {'003',30,1e-4,1e-3; '003',30,1e-3,1e-3; '003',100,1e-4,1e-3; '003',100,1e-3,1e-3; ...
%         '003',1000,1e-4,1e-3; '003',1000,1e-3,1e-3; '001',1000,1e-3,1e-3};
% stage 2: with the bound inactive (kappa 100) is the remaining swing loss the
% spatial prior beta? (E1_sens varied beta only at kappa 3, where the bound masked it)
% stage 2 (done, see README): runs = {'003',100,1e-3,1e-5; '003',100,1e-3,1e-4;
%   '003',100,1e-3,1e-3; '001',100,1e-3,1e-5};
% stage 3: the tidal control at beta 1e-4 (the beta that keeps the 003 swing)
runs = {'001', 100, 1e-3, 1e-4; '001', 100, 1e-3, 1e-3};

pl = 1;  fm = S.fmdl1;  Ne = size(fm.elems, 1);
lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
s0 = ones(Ne, 1);  s0(lung_idx) = sig_lung;
J  = calc_jacobian(mk_image(fm, s0));
grp = zeros(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
Gc = calc_dual_graph(fm, lung_idx, struct('target_N', N_coarse, 'groups', grp));
Ac = J * Gc.EL;
yl = Gc.cent(:,2);  r1 = yl > wmedian(yl, Gc.vol);          % r1 = anterior
L  = norm(sum(Gc.cent(r1,:).*Gc.vol(r1),1)/sum(Gc.vol(r1)) - ...
          sum(Gc.cent(~r1,:).*Gc.vol(~r1),1)/sum(Gc.vol(~r1)));

rt  = recs([recs.plane] == pl & strcmp({recs.id}, '001'));
Yt  = calc_difference_data(rt.vh_low, rt.V_lp(:, 1:ds:end), fm);
rgt = -gn_frames(Ac, Yt, lcurve_hp(Ac, Yt));
gt  = Gc.vol' * rgt;  [~, o] = sort(gt);  n10 = max(3, round(0.1*numel(gt)));
dtiv = mean(rgt(:, o(end-n10+1:end)), 2) - mean(rgt(:, o(1:n10)), 2);
w_tidal = max(dtiv, 0);  tiv_g = max(sum(Gc.vol .* dtiv) / sum(Gc.vol), 0);
fprintf('mean tidal TIV per unit volume (001): %.4g\n', tiv_g);

hsw = @(X) p2p((Gc.vol(r1)' * X(r1,:) - Gc.vol(~r1)' * X(~r1,:)) / 2);
cache = struct();
R = zeros(size(runs,1), 10);
for k = 1:size(runs, 1)
    id = runs{k,1};  kappa = runs{k,2};  alpha = runs{k,3};  beta = runs{k,4};
    if ~isfield(cache, ['r' id])
        r = recs([recs.plane] == pl & strcmp({recs.id}, id));
        if strcmp(r.kind, 'paradox'); vh = r.vh_mean; else; vh = r.vh_low; end
        Yd  = calc_difference_data(vh, r.V_lp(:, 1:ds:round(win_s*fps)), fm);
        rgn = -gn_frames(Ac, Yd, lcurve_hp(Ac, Yd));
        cache.(['r' id]) = struct('Yd', Yd, 'rgn', rgn);
        mn = min(rgn, [], 2);
        fprintf('%s: GN min gas change per node, in units of mean TIV: median %.2f, 5th pct %.2f, min %.2f\n', ...
                id, median(mn)/tiv_g, prc(mn, 5)/tiv_g, min(mn)/tiv_g);
    end
    Yd = cache.(['r' id]).Yd;  rgn = cache.(['r' id]).rgn;
    rho0 = kappa * tiv_g;
    o = struct('alpha', alpha, 'beta', beta, 'delta', dm*L/pi, 'gamma', 1e3, 'dt', dt, ...
               'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
               'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
    [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yd, Gc, o);
    QG   = hk_pendelluft(Gc, m, r1, dt);
    gsrc = dt * (Gc.vol' * info.zeta);
    swg  = hsw(rgn);
    act  = (rho + rho0) <= 1e-3 * rho0;
    R(k,:) = [kappa, alpha, hsw(rho)/max(swg,eps), hsw(max(rgn, -rho0))/max(swg,eps), ...
              norm(-Ac*rho - Yd, 'fro')/norm(Yd, 'fro'), norm(-Ac*rgn - Yd, 'fro')/norm(Yd, 'fro'), ...
              mean(act(:)), mean(mean(act(r1,:))), mean(mean(act(~r1,:))), ...
              sum(abs(dt*QG)) / (sum(abs(dt*QG)) + sum(abs(gsrc)) + eps)];
    fprintf(['%s kappa %4d alpha %.0e beta %.0e | swing HK/GN %.2f clipGN/GN %.2f | misfit HK %.3f GN %.3f | ' ...
             'bound all %.3f ant %.3f post %.3f | T-share %.2f | flux-PV %.3e | res %.0e | %d it\n'], ...
             id, kappa, alpha, beta, R(k,3), R(k,4), R(k,5), R(k,6), R(k,7), R(k,8), R(k,9), R(k,10), ...
             dt*sum(max(QG,0)), max(info.res), info.iters);
end
save(fullfile(here, 'diag_E1_pos_stage3.mat'), 'R', 'runs', 'tiv_g');

% ======================================================================
function v = p2p(x); v = max(x) - min(x); end

function q = prc(x, pc)
x = sort(x(:));  n = numel(x);  r = 1 + (n-1)*pc/100;  lo = floor(r);  hi = ceil(r);
q = x(lo) + (r - lo) * (x(hi) - x(lo));
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

function m = wmedian(x, w)
[xs, o] = sort(x);  cw = cumsum(w(o)) / sum(w);
m = xs(find(cw >= 0.5, 1));
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end
