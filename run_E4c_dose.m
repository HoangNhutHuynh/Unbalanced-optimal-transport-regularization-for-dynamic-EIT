%RUN_E4C_DOSE  E4c: independent validation of HK protocol P1 by dose-response.
%
%   P1 (alpha = 1e-3, delta = 4 L_AP/pi, rho0 = 3 * mean TIV, ventilation
%   mode with w = TIV map) was chosen after E4b. E4c tests it on NEW phantoms
%   with NO parameter sweep and NO ground truth anywhere in the HK pipeline:
%   the TIV map comes from a GN-cluster reconstruction with a FIXED hp.
%   Baselines keep a ground-truth oracle on rel err (favours them).
%
%   Phantom family (as E4b): soft regions A/P, rho = v u + Q dP, transfer
%   Q(t) = Q0 * bump(t). New vs E4b:
%     S1: different boundary tilt, transfer LATE in inspiration
%     S2: E4b tilt, transfer early, ventilation heterogeneity eta = 0.6 (x2)
%   Dose: Q0 / Q0max in {0, .25, .5, 1, 2, 3}  (0 = no pendelluft; < 1
%   masked: region A never loses volume; > 1 overt). Noise {0.5%, 1%},
%   2 seeds each.
%
%   Reported per scenario x noise: PV estimate vs PV_gt for HK flux-PV,
%   FRIC-vol and region-out on GN-cluster and L2-ST images; linear fit
%   (slope, intercept, R^2) over the doses; false positive at dose 0;
%   detection = smallest dose whose PV exceeds 3x the dose-0 value.
%
%   ~1 h. Needs EIDORS.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
if ~exist('mk_library_model','file'); error('EIDORS not on path.'); end

% ---------------- knobs ----------------
doses  = [0 0.25 0.5 1 2 3];
noises = [0.005 0.01];
seeds  = [11 12];
scen = struct( ...
  'name', {'S1 tilt-B, late transfer', 'S2 strong heterogeneity'}, ...
  'tilt', {[-0.40 -1.00 -0.30],        [0.30 -1.00 0.25]}, ...
  'kq',   {8:14,                        2:10}, ...
  'eta',  {0.3,                         0.6});
fps = 10;  dt = 1/fps;  T1 = 31;  k_ins = 1:16;  Vt = 0.10;  ell = 0.04;
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;
P1 = struct('alpha', 1e-3, 'dm', 4, 'kappa', 3, 'beta', 1e-2);
hp_w = 1e-1;                     % FIXED GN hp for the TIV map (E4: GN best)

% ---------------- model + graphs ----------------
fmdl = mk_library_model('adult_male_16el_lungs');
fmdl.stimulation = mk_stim_patterns(16,1,[0,1],[0,1],{'no_meas_current'},1);
fmdl = mdl_normalize(fmdl, 0);
Ne = size(fmdl.elems,1);
lung_idx = [fmdl.mat_idx{2}(:); fmdl.mat_idx{3}(:)];
sigma0 = ones(Ne,1);  sigma0(lung_idx) = sig_lung;
G  = calc_dual_graph(fmdl, lung_idx);
grp = zeros(Ne,1);  grp(fmdl.mat_idx{2}) = 1;  grp(fmdl.mat_idx{3}) = 2;
Gc = calc_dual_graph(fmdl, lung_idx, struct('target_N', N_coarse, 'groups', grp));
imdl = eidors_obj('inv_model', 'E4c');
imdl.fwd_model = fmdl;  imdl.reconst_type = 'difference';
imdl.jacobian_bkgnd.value = sigma0;
imdl.RtR_prior = @prior_noser;  imdl.hyperparameter.value = 1e-2;
Jf = calc_jacobian(calc_jacobian_bkgnd(imdl));
Ac = Jf * Gc.EL;
img0 = mk_image(fmdl, sigma0);  vh = fwd_solve(img0);  VH = vh.meas;
ins_steps = (1:T1-1)' < k_ins(end);

meth = {'HK P1 flux-PV', 'FRIC-vol (GN)', 'FRIC-vol (L2-ST)', 'region-out (GN)', 'region-out (L2-ST)'};
ALL = struct();
for is = 1:numel(scen)
    sc = scen(is);
    P = make_phantom(G, Gc, sc, T1, k_ins, Vt, ell);
    dref = P.delta_ref;
    fprintf('\n######## %s: Q0max %.5f, delta_ref %.3f, gamma %.3f, straddling clusters %d\n', ...
            sc.name, P.Q0max, dref, P.gamma, P.nmixed);
    for in = 1:numel(noises)
        nl = noises(in);
        tab = nan(numel(doses), numel(seeds), 1 + numel(meth));   % [PV_gt, methods...]
        err = nan(numel(doses), numel(seeds), 3);                  % rel err HK, GN, L2
        for id = 1:numel(doses)
            Q0 = doses(id) * P.Q0max;
            rho_true = P.u * P.v + P.dP * (Q0 * P.qsh);
            ds_true  = -(G.EL * rho_true);
            PV_gt    = Q0 * P.gamma;
            rel_lung = @(DS) norm(DS(lung_idx,:) - ds_true(lung_idx,:), 'fro') / ...
                             norm(ds_true(lung_idx,:), 'fro');
            % noiseless data once per dose
            VI0 = zeros(numel(VH), T1);
            for k = 1:T1
                img0.elem_data = sigma0 + ds_true(:,k);
                vv = fwd_solve(img0);  VI0(:,k) = vv.meas;
            end
            for ir = 1:numel(seeds)
                rs = RandStream('mt19937ar', 'Seed', seeds(ir) + 100*id + 1000*in + 10000*is);
                VI = VI0 + nl * (norm(VI0 - VH, 'fro')/sqrt(numel(VI0))) * randn(rs, size(VI0));
                Yd = calc_difference_data(VH, VI, fmdl);

                % ---- HK P1: no ground truth anywhere ----
                rho_gn = -gn_frames(Ac, Yd, hp_w);
                tiv = max(rho_gn(:, k_ins(end)) - rho_gn(:, 1), 0);
                tiv_m = sum(Gc.vol .* tiv) / sum(Gc.vol);
                o = struct('alpha', P1.alpha, 'delta', P1.dm*dref, 'beta', P1.beta, ...
                           'gamma', 1e3, 'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, ...
                           'verbose', 0, 'use_eta', false, 'normalize', true, ...
                           'tau0', tau0, 'rho0', P1.kappa*tiv_m, 'vent_w', tiv);
                [rho, m, ~, info] = hk_core(-Jf, speye(size(Jf,1)), Yd, Gc, o);
                [~, pv_hk] = hk_pendelluft(Gc, m, P.hardP_c, dt, ins_steps);
                er_hk = rel_lung(-(Gc.EL*rho));

                % ---- baselines (oracle on rel err) ----
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
                pg = posthoc(best_gn.x, G, P.hardP, ins_steps);
                pl = posthoc(best_l2.x, G, P.hardP, ins_steps);

                tab(id, ir, :) = [PV_gt, pv_hk, pg(1), pl(1), pg(2), pl(2)];
                err(id, ir, :) = [er_hk, best_gn.e, best_l2.e];
                fprintf(['  noise %.1f%% dose %.2f seed %d | PV_gt %.5f | HK %.5f (err %.3f, res %.0e) | ' ...
                         'FRIC GN %.5f L2 %.5f | reg GN %.5f L2 %.5f\n'], 100*nl, doses(id), ir, ...
                         PV_gt, pv_hk, er_hk, max(info.res), pg(1), pl(1), pg(2), pl(2));
            end
        end
        % ---- summary for this scenario x noise ----
        fprintf('\n  === %s, noise %.1f%% (mean over seeds) ===\n', sc.name, 100*nl);
        fprintf('  %-20s %8s %8s %7s %12s %10s\n', 'estimator', 'slope', 'icpt', 'R^2', 'FP(dose 0)', 'detect>=');
        pv_gt = mean(tab(:,:,1), 2);
        for j = 1:numel(meth)
            y  = mean(tab(:,:,1+j), 2);
            cf = polyfit(pv_gt, y, 1);
            r2 = 1 - sum((y - polyval(cf, pv_gt)).^2) / max(sum((y - mean(y)).^2), eps);
            fp = y(1);
            dd = find(y(2:end) > 3*max(fp, 1e-6), 1);
            if isempty(dd); dstr = 'none'; else; dstr = sprintf('%.2f', doses(dd+1)); end
            fprintf('  %-20s %8.3f %8.5f %7.3f %12.5f %10s\n', meth{j}, cf(1), cf(2), r2, fp, dstr);
        end
        fprintf('  rel err (mean over doses/seeds): HK %.4f | GN %.4f | L2-ST %.4f\n', ...
                mean(err(:,:,1), 'all'), mean(err(:,:,2), 'all'), mean(err(:,:,3), 'all'));
        key = sprintf('s%d_n%d', is, in);
        ALL.(key) = struct('scen', sc.name, 'noise', nl, 'tab', tab, 'err', err, ...
                           'doses', doses, 'meth', {meth});
    end
end
save E4c_results.mat ALL doses noises seeds scen P1 hp_w

% ---------------- figure: dose-response ----------------
figure('Name', 'E4c dose-response');
keys = fieldnames(ALL);
for q = 1:numel(keys)
    S = ALL.(keys{q});
    subplot(2, 2, q);  hold on;
    x = mean(S.tab(:,:,1), 2);
    for j = 1:numel(meth)
        plot(x, mean(S.tab(:,:,1+j), 2), '-o');
    end
    plot(x, x, 'k--');  hold off;
    xlabel('PV ground truth');  ylabel('PV estimate');
    title(sprintf('%s, noise %.1f%%', S.scen, 100*S.noise));
end
legend([meth, {'identity'}], 'Location', 'northwest');

% ======================================================================
function P = make_phantom(G, Gc, sc, T1, k_ins, Vt, ell)
n   = sc.tilt / norm(sc.tilt);
xc  = sum(G.cent .* G.vol, 1) / sum(G.vol);
s   = (G.cent - xc) * n';
s0  = wmedian(s, G.vol);
phiP = 1 ./ (1 + exp(-(s - s0)/ell));  phiA = 1 - phiP;
VP = sum(G.vol .* phiP);  VA = sum(G.vol .* phiA);
P.hardP   = phiP > 0.5;
P.hardP_c = 1 ./ (1 + exp(-(((Gc.cent - xc) * n') - s0)/ell)) > 0.5;
P.nmixed  = nnz(accumarray(Gc.cluster, P.hardP, [], @(x) any(x) && ~all(x)));
P.dP      = phiP/VP - phiA/VA;
P.gamma   = sum(G.vol(P.hardP) .* P.dP(P.hardP));
v = zeros(1, T1);
v(k_ins) = Vt * sin(pi/2 * (k_ins-1)/(numel(k_ins)-1));
k_exp = k_ins(end):T1;
v(k_exp) = Vt * cos(pi/2 * (k_exp-k_ins(end))/(T1-k_ins(end))).^2;
P.v = v;
u = 1 + sc.eta*phiP;  P.u = u * sum(G.vol) / sum(G.vol .* u);
q = zeros(1, T1);  kq = sc.kq;  q(kq) = sin(pi*(kq-kq(1))/(kq(end)-kq(1))).^2;
P.qsh = q;
aA = sum(G.vol(~P.hardP) .* P.u(~P.hardP));
gA = -sum(G.vol(~P.hardP) .* P.dP(~P.hardP));
dv = diff(v(k_ins));  dq = diff(q(k_ins));
P.Q0max = min(aA*dv(dq > 0) ./ (gA*dq(dq > 0)));
cP = sum(G.cent(P.hardP,:).*G.vol(P.hardP),1)/sum(G.vol(P.hardP));
cA = sum(G.cent(~P.hardP,:).*G.vol(~P.hardP),1)/sum(G.vol(~P.hardP));
P.delta_ref = norm(cP - cA) / pi;
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
