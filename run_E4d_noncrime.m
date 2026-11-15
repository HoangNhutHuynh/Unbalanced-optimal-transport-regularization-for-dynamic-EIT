%RUN_E4D_NONCRIME  E4d: E4c dose-response WITHOUT the FEM inverse crime.
%
%   Data are simulated on a FINER, different forward mesh (maxh 0.05, same
%   thorax/lung geometry and electrodes, built with mk_library_model) and
%   reconstructed on the standard model (maxh 0.08; Jacobian, cluster graph
%   and HK protocol P1 exactly as in E4c - no sweep, no ground truth in the
%   HK pipeline, TIV map from GN-cluster with a fixed hp).
%
%   The phantom is defined CONTINUOUSLY in space: all constants (centre,
%   weighted-median threshold, |A|, |P|, normalization of u, gamma, Q0max,
%   delta_ref) are computed on the fine mesh (closest to the continuum) and
%   the fields are evaluated at element centroids of BOTH meshes: fine ->
%   data, reconstruction mesh -> ground truth for rel err and for the
%   image-based estimators. PV_gt = Q0 * gamma (fine mesh).
%
%   Scenarios: S0 = E4b geometry (early transfer, eta 0.3), S1, S2 = E4c.
%   Doses {0 .25 .5 1 2 3} x Q0max, noise 1%, 2 seeds. ~1 h (+ one netgen
%   build of the fine mesh, cached by mk_library_model).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
if ~exist('mk_library_model','file'); error('EIDORS not on path.'); end

% ---------------- knobs ----------------
doses  = [0 0.25 0.5 1 2 3];
nl     = 0.01;
seeds  = [21 22];
scen = struct( ...
  'name', {'S0 E4b geometry', 'S1 tilt-B, late transfer', 'S2 strong heterogeneity'}, ...
  'tilt', {[0.30 -1.00 0.25],  [-0.40 -1.00 -0.30],        [0.30 -1.00 0.25]}, ...
  'kq',   {2:10,               8:14,                        2:10}, ...
  'eta',  {0.3,                0.3,                         0.6});
fps = 10;  dt = 1/fps;  T1 = 31;  k_ins = 1:16;  Vt = 0.10;  ell = 0.04;
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;
P1 = struct('alpha', 1e-3, 'dm', 4, 'kappa', 3, 'beta', 1e-2);
hp_w = 1e-1;
maxh_fwd = 0.05;                 % reconstruction mesh: library default 0.08

% ---------------- reconstruction model (standard) ----------------
fmdl = mk_library_model('adult_male_16el_lungs');
stim = mk_stim_patterns(16,1,[0,1],[0,1],{'no_meas_current'},1);
fmdl.stimulation = stim;  fmdl = mdl_normalize(fmdl, 0);
Ne = size(fmdl.elems,1);
lung_idx = [fmdl.mat_idx{2}(:); fmdl.mat_idx{3}(:)];
sigma0 = ones(Ne,1);  sigma0(lung_idx) = sig_lung;
G  = calc_dual_graph(fmdl, lung_idx);
grp = zeros(Ne,1);  grp(fmdl.mat_idx{2}) = 1;  grp(fmdl.mat_idx{3}) = 2;
Gc = calc_dual_graph(fmdl, lung_idx, struct('target_N', N_coarse, 'groups', grp));
imdl = eidors_obj('inv_model', 'E4d');
imdl.fwd_model = fmdl;  imdl.reconst_type = 'difference';
imdl.jacobian_bkgnd.value = sigma0;
imdl.RtR_prior = @prior_noser;  imdl.hyperparameter.value = 1e-2;
Jf = calc_jacobian(calc_jacobian_bkgnd(imdl));
Ac = Jf * Gc.EL;

% ---------------- forward model (finer, different mesh) ----------------
ff = mk_library_model({'adult_male','boundary','left_lung','right_lung'}, ...
                      [16 1 0.5], [0.05], maxh_fwd);
ff.stimulation = stim;  ff = mdl_normalize(ff, 0);
Nf = size(ff.elems,1);
lung_f = [ff.mat_idx{2}(:); ff.mat_idx{3}(:)];
sig0_f = ones(Nf,1);  sig0_f(lung_f) = sig_lung;
cf_all = interp_mesh(ff, 0);  vf_all = get_elem_volume(ff);
cf = cf_all(lung_f,:);  vf = vf_all(lung_f);
ep_r = elec_xy(fmdl);  ep_f = elec_xy(ff);
fprintf(['forward mesh: %d elems (%d lung) | reconstruction mesh: %d elems (%d lung)\n' ...
         'max electrode-centre mismatch between meshes: %.3g\n'], Nf, numel(lung_f), ...
         Ne, numel(lung_idx), max(sqrt(sum((ep_r - ep_f).^2, 2))));
img_f = mk_image(ff, sig0_f);  vh = fwd_solve(img_f);  VH = vh.meas;
ins_steps = (1:T1-1)' < k_ins(end);
cr_lung = G.cent;                                  % reconstruction lung nodes

meth = {'HK P1 flux-PV', 'FRIC-vol (GN)', 'FRIC-vol (L2-ST)', 'region-out (GN)', 'region-out (L2-ST)'};
ALL = struct();
for is = 1:numel(scen)
    sc = scen(is);
    C  = phantom_consts(cf, vf, sc, ell, T1, k_ins, Vt);
    [u_f, dP_f]          = phantom_fields(cf, C);
    [u_r, dP_r, phiP_r]  = phantom_fields(cr_lung, C);
    [~, ~, phiP_c]       = phantom_fields(Gc.cent, C);
    hardP = phiP_r > 0.5;  hardP_c = phiP_c > 0.5;
    fprintf('\n######## %s: Q0max %.5f, delta_ref %.3f, gamma %.3f (fine mesh)\n', ...
            sc.name, C.Q0max, C.delta_ref, C.gamma);

    tab = nan(numel(doses), numel(seeds), 1 + numel(meth));
    err = nan(numel(doses), numel(seeds), 3);
    gte = nan(numel(doses), 2);                    % image estimators on GT images
    for id = 1:numel(doses)
        Q0 = doses(id) * C.Q0max;
        rho_f = u_f * C.v + dP_f * (Q0 * C.qsh);            % fine lung elements
        rho_r = u_r * C.v + dP_r * (Q0 * C.qsh);            % reconstruction nodes
        ds_true = -(G.EL * rho_r);                          % GT on the reconstruction mesh
        PV_gt   = Q0 * C.gamma;
        rel_lung = @(DS) norm(DS(lung_idx,:) - ds_true(lung_idx,:), 'fro') / ...
                         norm(ds_true(lung_idx,:), 'fro');
        gte(id,:) = posthoc(ds_true, G, hardP, ins_steps);
        VI0 = zeros(numel(VH), T1);
        for k = 1:T1
            img_f.elem_data = sig0_f;
            img_f.elem_data(lung_f) = sig_lung - rho_f(:,k);
            vv = fwd_solve(img_f);  VI0(:,k) = vv.meas;
        end
        for ir = 1:numel(seeds)
            rs = RandStream('mt19937ar', 'Seed', seeds(ir) + 100*id + 10000*is);
            VI = VI0 + nl * (norm(VI0 - VH, 'fro')/sqrt(numel(VI0))) * randn(rs, size(VI0));
            Yd = calc_difference_data(VH, VI, fmdl);

            rho_gn = -gn_frames(Ac, Yd, hp_w);
            tiv = max(rho_gn(:, k_ins(end)) - rho_gn(:, 1), 0);
            tiv_m = sum(Gc.vol .* tiv) / sum(Gc.vol);
            o = struct('alpha', P1.alpha, 'delta', P1.dm*C.delta_ref, 'beta', P1.beta, ...
                       'gamma', 1e3, 'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, ...
                       'verbose', 0, 'use_eta', false, 'normalize', true, ...
                       'tau0', tau0, 'rho0', P1.kappa*tiv_m, 'vent_w', tiv);
            [rho, m, ~, info] = hk_core(-Jf, speye(size(Jf,1)), Yd, Gc, o);
            [~, pv_hk] = hk_pendelluft(Gc, m, hardP_c, dt, ins_steps);
            er_hk = rel_lung(-(Gc.EL*rho));

            best_gn = struct('e', Inf);
            for hp = logspace(-3, 0, 7)
                x = Gc.EL * gn_frames(Ac, Yd, hp);  e = rel_lung(x);
                if e < best_gn.e; best_gn = struct('e', e, 'x', x); end
            end
            best_l2 = struct('e', Inf);
            for ls = [1e-3 1e-2 1e-1]
                for lt = [1e-4 1e-3 1e-2]
                    x = Gc.EL * l2st(Ac, Yd, ls, lt);  e = rel_lung(x);
                    if e < best_l2.e; best_l2 = struct('e', e, 'x', x); end
                end
            end
            pg = posthoc(best_gn.x, G, hardP, ins_steps);
            pl = posthoc(best_l2.x, G, hardP, ins_steps);
            tab(id, ir, :) = [PV_gt, pv_hk, pg(1), pl(1), pg(2), pl(2)];
            err(id, ir, :) = [er_hk, best_gn.e, best_l2.e];
            fprintf(['  dose %.2f seed %d | PV_gt %.5f | HK %.5f (err %.3f, res %.0e) | ' ...
                     'FRIC GN %.5f L2 %.5f | reg GN %.5f L2 %.5f\n'], doses(id), ir, ...
                     PV_gt, pv_hk, er_hk, max(info.res), pg(1), pl(1), pg(2), pl(2));
        end
    end
    fprintf('\n  === %s, noise %.1f%%, NO inverse crime (mean over seeds) ===\n', sc.name, 100*nl);
    fprintf('  %-20s %8s %8s %7s %12s %10s\n', 'estimator', 'slope', 'icpt', 'R^2', 'FP(dose 0)', 'FP/PV(1)');
    pv_gt = mean(tab(:,:,1), 2);
    i1 = find(doses == 1, 1);
    for j = 1:numel(meth)
        y  = mean(tab(:,:,1+j), 2);
        cfit = polyfit(pv_gt, y, 1);
        r2 = 1 - sum((y - polyval(cfit, pv_gt)).^2) / max(sum((y - mean(y)).^2), eps);
        fprintf('  %-20s %8.3f %8.5f %7.3f %12.5f %10.3f\n', meth{j}, cfit(1), cfit(2), r2, ...
                y(1), y(1) / max(y(i1), eps));
    end
    fprintf('  image estimators on GT images (reconstruction mesh), dose 1 / dose 3: FRIC %.5f / %.5f, region-out %.5f / %.5f (PV_gt %.5f / %.5f)\n', ...
            gte(i1,1), gte(end,1), gte(i1,2), gte(end,2), pv_gt(i1), pv_gt(end));
    fprintf('  rel err (mean): HK %.4f | GN %.4f | L2-ST %.4f\n', ...
            mean(err(:,:,1), 'all'), mean(err(:,:,2), 'all'), mean(err(:,:,3), 'all'));
    ALL.(sprintf('s%d', is)) = struct('scen', sc.name, 'tab', tab, 'err', err, 'gte', gte, ...
                                      'doses', doses, 'meth', {meth}, 'C', C);
end
save E4d_results.mat ALL doses nl seeds scen P1 hp_w maxh_fwd

figure('Name', 'E4d dose-response (no inverse crime)');
keys = fieldnames(ALL);
for q = 1:numel(keys)
    S = ALL.(keys{q});
    subplot(1, numel(keys), q);  hold on;
    x = mean(S.tab(:,:,1), 2);
    for j = 1:numel(meth); plot(x, mean(S.tab(:,:,1+j), 2), '-o'); end
    plot(x, x, 'k--');  hold off;
    xlabel('PV ground truth');  ylabel('PV estimate');  title(S.scen);
end
legend([meth, {'identity'}], 'Location', 'northwest');

% ======================================================================
function C = phantom_consts(cf, vf, sc, ell, T1, k_ins, Vt)
% all phantom constants from the FINE mesh (lung element centroids cf,
% volumes vf)
C.n   = sc.tilt / norm(sc.tilt);  C.ell = ell;  C.eta = sc.eta;
C.xc  = sum(cf .* vf, 1) / sum(vf);
s     = (cf - C.xc) * C.n';
C.s0  = wmedian(s, vf);
phiP  = 1 ./ (1 + exp(-(s - C.s0)/ell));
C.VP  = sum(vf .* phiP);   C.VA = sum(vf .* (1 - phiP));
u     = 1 + sc.eta*phiP;   C.unorm = sum(vf) / sum(vf .* u);
v = zeros(1, T1);
v(k_ins) = Vt * sin(pi/2 * (k_ins-1)/(numel(k_ins)-1));
k_exp = k_ins(end):T1;
v(k_exp) = Vt * cos(pi/2 * (k_exp-k_ins(end))/(T1-k_ins(end))).^2;
C.v = v;
q = zeros(1, T1);  kq = sc.kq;  q(kq) = sin(pi*(kq-kq(1))/(kq(end)-kq(1))).^2;
C.qsh = q;
hard = phiP > 0.5;
dP   = phiP/C.VP - (1 - phiP)/C.VA;
C.gamma = sum(vf(hard) .* dP(hard));
uu = C.unorm * u;
aA = sum(vf(~hard) .* uu(~hard));
gA = -sum(vf(~hard) .* dP(~hard));
dv = diff(v(k_ins));  dq = diff(q(k_ins));
C.Q0max = min(aA*dv(dq > 0) ./ (gA*dq(dq > 0)));
cP = sum(cf(hard,:) .* vf(hard), 1) / sum(vf(hard));
cA = sum(cf(~hard,:) .* vf(~hard), 1) / sum(vf(~hard));
C.delta_ref = norm(cP - cA) / pi;
end

function [u, dP, phiP] = phantom_fields(c, C)
s    = (c - C.xc) * C.n';
phiP = 1 ./ (1 + exp(-(s - C.s0)/C.ell));
u    = C.unorm * (1 + C.eta*phiP);
dP   = phiP/C.VP - (1 - phiP)/C.VA;
end

function ep = elec_xy(fm)
ne = numel(fm.electrode);  ep = zeros(ne, size(fm.nodes,2));
for k = 1:ne; ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes,:), 1); end
end

function pe = posthoc(DS, G, hardP, ins_steps)
rho = -(G.EL' * DS);
d   = diff(rho, 1, 2);  d = d(:, ins_steps);
fric = sum(sum(G.vol .* max(-d, 0)));
dVA  = G.vol(~hardP)' * d(~hardP, :);
pe = [fric, sum(max(-dVA, 0))];
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

function X = l2st(A, Y, ls, lt)
[n, T1] = deal(size(A,2), size(Y,2));
H   = A' * A;
Rs  = diag(sqrt(diag(H)));
V   = cos(pi * ((0:T1-1)' + 0.5) * (0:T1-1) / T1);
V   = V ./ sqrt(sum(V.^2, 1));
lam = 2 - 2*cos(pi * (0:T1-1) / T1);
B   = (A' * Y) * V;
Z   = zeros(n, T1);
for k = 1:T1
    Rk = chol(H + ls*Rs + (lt*lam(k))*eye(n));
    Z(:,k) = Rk \ (Rk' \ B(:,k));
end
X = Z * V';
end
