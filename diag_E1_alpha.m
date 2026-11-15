%DIAG_E1_ALPHA  Why does HK shrink the anti-phase swing on real data?
%
%   run_E1_sens: lowering beta 100x does NOT restore the GN swing in the
%   paradoxical record (1pl 003: HK/GN swing 0.25 / 0.24 / 0.24), while the
%   tidal record keeps it (1.06-1.34). Two candidate mechanisms:
%     (a) regularization shrinkage: the in-phase component is free (vent
%         mode c), the anti-phase one costs transport/Hellinger action; the
%         data term (alpha) is too weak to keep it  -> ratio rises with alpha
%     (b) positivity: rho + rho0 >= 0 clips the deflating region when the
%         anti-phase amplitude exceeds rho0        -> ratio rises with kappa
%   Grid (1-plane, anterior|posterior, beta 1e-3, records 001 and 003):
%     alpha in {1e-3, 1e-2, 1e-1} x kappa in {3, 30}.
%   Per run: swing ratio HK/GN, relative data misfit HK and GN, fraction of
%   (node, frame) at the positivity bound, T-share.
%   Also: L-curve hp and edge flag for all 4 (model, record) GN references.
%
%   ~15 min. Needs E1_prep.mat and EIDORS.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
S = load(fullfile(here, 'E1_prep.mat'));
recs = S.recs_save;  fps = S.fps;

alphas = [1e-3 1e-2 1e-1];  kappas = [3 30];  beta = 1e-3;
ids = {'001', '003'};  win_s = 20;  ds = 2;  dt = ds / fps;
dm = 4;  sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;
models = {S.fmdl1, S.fmdl2};

% ---------------- GN references: hp and L-curve edge flag ----------------
fprintf('---- GN reference (L-curve) ----\n');
GD = cell(2, 1);
for pl = 1:2
    fm = models{pl};  Ne = size(fm.elems, 1);
    lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
    s0 = ones(Ne, 1);  s0(lung_idx) = sig_lung;
    J  = calc_jacobian(mk_image(fm, s0));
    grp = zeros(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
    Gc = calc_dual_graph(fm, lung_idx, struct('target_N', N_coarse, 'groups', grp));
    Ac = J * Gc.EL;
    GD{pl} = struct('J', J, 'Gc', Gc, 'Ac', Ac);
    for ii = 1:numel(ids)
        r  = recs([recs.plane] == pl & strcmp({recs.id}, ids{ii}));
        if strcmp(r.kind, 'paradox'); vh = r.vh_mean; else; vh = r.vh_low; end
        Yd = calc_difference_data(vh, r.V_lp(:, 1:ds:round(win_s*fps)), fm);
        [hp, edge, lr] = lcurve_hp(Ac, Yd);
        fprintf('%dpl %s: hp %.3g | corner at the %s of the scanned range %s\n', pl, ids{ii}, hp, ...
                ternary(edge, 'EDGE', 'interior'), sprintf('[%.1e %.1e]', lr(1), lr(2)));
    end
    if pl == 1; clear J; end
end

% ---------------- alpha x kappa grid, 1-plane ----------------
pl = 1;  fm = models{pl};  J = GD{pl}.J;  Gc = GD{pl}.Gc;  Ac = GD{pl}.Ac;
yl = Gc.cent(:,2);  r1 = yl > wmedian(yl, Gc.vol);
L  = norm(sum(Gc.cent(r1,:).*Gc.vol(r1),1)/sum(Gc.vol(r1)) - ...
          sum(Gc.cent(~r1,:).*Gc.vol(~r1),1)/sum(Gc.vol(~r1)));
rt  = recs([recs.plane] == pl & strcmp({recs.id}, '001'));
Yt  = calc_difference_data(rt.vh_low, rt.V_lp(:, 1:ds:end), fm);
rgt = -gn_frames(Ac, Yt, lcurve_hp(Ac, Yt));
gt  = Gc.vol' * rgt;  [~, o] = sort(gt);  n10 = max(3, round(0.1*numel(gt)));
dtiv = mean(rgt(:, o(end-n10+1:end)), 2) - mean(rgt(:, o(1:n10)), 2);
w_tidal = max(dtiv, 0);  tiv_g = max(sum(Gc.vol .* dtiv) / sum(Gc.vol), 0);

fprintf('\n---- 1-plane, anterior|posterior, beta %.0e ----\n', beta);
R = [];
for ii = 1:numel(ids)
    r = recs([recs.plane] == pl & strcmp({recs.id}, ids{ii}));
    if strcmp(r.kind, 'paradox'); vh = r.vh_mean; else; vh = r.vh_low; end
    Yd  = calc_difference_data(vh, r.V_lp(:, 1:ds:round(win_s*fps)), fm);
    hp  = lcurve_hp(Ac, Yd);
    rgn = -gn_frames(Ac, Yd, hp);
    mis_gn = norm(-Ac*rgn - Yd, 'fro') / norm(Yd, 'fro');
    swg = p2p((Gc.vol(r1)' * rgn(r1,:) - Gc.vol(~r1)' * rgn(~r1,:)) / 2);
    for kappa = kappas
        rho0 = kappa * tiv_g;
        for alpha = alphas
            o = struct('alpha', alpha, 'beta', beta, 'delta', dm*L/pi, 'gamma', 1e3, 'dt', dt, ...
                       'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
                       'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w_tidal);
            [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yd, Gc, o);
            QG   = hk_pendelluft(Gc, m, r1, dt);
            gsrc = dt * (Gc.vol' * info.zeta);
            swh  = p2p((Gc.vol(r1)' * rho(r1,:) - Gc.vol(~r1)' * rho(~r1,:)) / 2);
            mis  = norm(-Ac*rho - Yd, 'fro') / norm(Yd, 'fro');
            act  = mean(rho(:) + rho0 <= 1e-3 * rho0);
            Tsh  = sum(abs(dt*QG)) / (sum(abs(dt*QG)) + sum(abs(gsrc)) + eps);
            fprintf(['1pl %s kappa %2d alpha %.0e | swing HK/GN %.2f | misfit HK %.3f GN %.3f | ' ...
                     'at positivity bound %.3f | T-share %.2f | flux-PV %.3e | res %.0e | %d it\n'], ...
                     ids{ii}, kappa, alpha, swh/max(swg,eps), mis, mis_gn, act, Tsh, ...
                     dt*sum(max(QG,0)), max(info.res), info.iters);
            R = [R; ii, kappa, alpha, swh/max(swg,eps), mis, mis_gn, act, Tsh]; %#ok<AGROW>
        end
    end
end
save(fullfile(here, 'diag_E1_alpha.mat'), 'R', 'alphas', 'kappas', 'beta');

fprintf('\n---- reading guide ----\n');
fprintf('ratio(003) rises as alpha DECREASES (alpha weights the action) -> (a) shrinkage\n');
fprintf('ratio(003) rises with kappa, bound fraction drops -> (b) positivity clipping\n');
fprintf('misfit HK >> misfit GN in 003 but not in 001     -> HK under-fits the anti-phase signal\n');

% ======================================================================
function v = p2p(x); v = max(x) - min(x); end

function s = ternary(c, a, b); if c; s = a; else; s = b; end; end

function [hp, edge, lr] = lcurve_hp(A, Y)
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
edge = k <= 6 || k >= numel(lams) - 5;
lr = sqrt(lams([1 end]));
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
