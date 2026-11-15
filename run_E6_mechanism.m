%RUN_E6_MECHANISM  E6: what does each pendelluft estimator measure when the
%   SAME regional gas field can come from different mechanisms?
%
%   Question (paper, identifiability): EIT sees only the gas field rho(x,t).
%   Gas moving between regions (transfer) and regions filling at different
%   times from the airway (time-constant asynchrony, the classic Otis
%   mechanism of pendelluft) both make regional volume curves that are out
%   of phase. Here the mechanism and the ground truth are known, and every
%   estimator is run on the same simulated data:
%     HK raw / HK quadrature (interface flux), and on GN images: opposite-flow
%     volume, quadrature on regional volumes (quadGN), FRIC-vol, region-out,
%     Sang pixel amplitude, regional lag (pend_estimators.m).
%   Ground truths from the TRUE gas field on the reconstruction mesh:
%     GT_opp = opposite-flow volume, GT_q = quadrature volume, true lag.
%
%   Model, meshes, phantom geometry, noise and HK protocol P1 exactly as E4d
%   (no inverse crime: data on a finer mesh, maxh 0.05). Region split A/P by
%   the E4 S0 tilt (soft partition phiP, TIV heterogeneity eta 0.3).
%   Time: 3 identical breaths (period 3.1 s, inspiration 1.5 s, 10 fps), all
%   courses periodic so delays are exact (sub-frame) and RC is steady state:
%     rho = u (1-phiP) vA(t) + u phiP vP(t) + dP Q0 q(t)
%   Conditions:
%     M0 in-phase only (vP = vA, Q0 = 0)                    -> GT 0
%     M1 transfer, Q0 = 0.5, 1, 2 x Q0max (E4 pulse in inspiration; up to
%        1 x Q0max region A never empties, so GT_opp = 0 by construction)
%     M2 delay, vP(t) = vA(t - tau), tau 50 / 100 / 200 ms
%     M3 RC, vP = vA through 1/(1 + i 2 pi f tau), tau 50 / 100 / 200 ms
%     M4 delay 100 ms + transfer 1 x Q0max
%   Reading:
%     an estimator that tracks GT across M1-M4 measures pendelluft whatever
%     its cause; one that tracks only M1 or only M2/M3 is mechanism-specific;
%     M0 gives each estimator's in-phase false positive.
%   Self-checks (oracle, true images): M0 quadGN = 0; M2 true lag in (0.5, 1.2) x tau
%   (first run: 39.8 ms at tau 50 ms - the soft A/P partition dilutes the regional lag).
%   MODE 'full': 2 seeds x 11 conditions = 22 HK runs of 93 frames, ~45 min -> E6_results.mat.
%   MODE 'noise' (2026-11-13): 4 conditions x 3 noise levels x 3 seeds = 36 runs, ~1 h ->
%   E6_noise_results.mat (does HK's lower floor on noisy nulls, E7, become better accuracy?).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
if ~exist('mk_library_model', 'file'); error('EIDORS not on path.'); end
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end

% ---------------- knobs ----------------
fps = 10;  dt = 1/fps;  T1 = 31;  NB = 3;  T = NB*T1;  Ti = 1.5;  Tb = T1*dt;
k_ins = 1:16;  Vt = 0.10;  ell = 0.04;  seeds = [61 62];
sc = struct('tilt', [0.30 -1.00 0.25], 'kq', 2:10, 'eta', 0.3);     % E4 S0 geometry
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;  hp_w = 1e-1;  maxh_fwd = 0.05;
P1 = struct('alpha', 1e-3, 'dm', 4, 'kappa', 3, 'beta', 1e-2);
COND = struct('name', {'M0 in-phase', 'M1 transfer 0.5', 'M1 transfer 1', 'M1 transfer 2', ...
                       'M2 delay 50 ms', 'M2 delay 100 ms', 'M2 delay 200 ms', ...
                       'M3 RC 50 ms', 'M3 RC 100 ms', 'M3 RC 200 ms', 'M4 delay 100 + transfer 1'}, ...
              'mech', {'none', 'none', 'none', 'none', 'delay', 'delay', 'delay', 'rc', 'rc', 'rc', 'delay'}, ...
              'tau',  {0, 0, 0, 0, 0.05, 0.10, 0.20, 0.05, 0.10, 0.20, 0.10}, ...
              'dose', {0, 0.5, 1, 2, 0, 0, 0, 0, 0, 0, 1});
MODE = 'full';    % 'full': all 11 conditions at 1 % noise -> E6_results.mat (done 2026-11-12, Fig14)
                  % 'noise': M0, transfer 0.5, delay 100, RC 100 x noise 0.3 / 3 / 10 % x 3 seeds -> E6_noise_results.mat
switch MODE
    case 'full';  NLS = 0.01;  CSEL = 1:numel(COND);  ofile = 'E6_results.mat';
    case 'noise'; NLS = [0.003 0.03 0.1];  CSEL = [1 2 6 9];  seeds = [61 62 63];  ofile = 'E6_noise_results.mat';
end
COND = COND(CSEL);

% ---------------- reconstruction model (as E4d) ----------------
fmdl = mk_library_model('adult_male_16el_lungs');
stim = mk_stim_patterns(16,1,[0,1],[0,1],{'no_meas_current'},1);
fmdl.stimulation = stim;  fmdl = mdl_normalize(fmdl, 0);
Ne = size(fmdl.elems,1);
lung_idx = [fmdl.mat_idx{2}(:); fmdl.mat_idx{3}(:)];
sigma0 = ones(Ne,1);  sigma0(lung_idx) = sig_lung;
G  = calc_dual_graph(fmdl, lung_idx);
grp = zeros(Ne,1);  grp(fmdl.mat_idx{2}) = 1;  grp(fmdl.mat_idx{3}) = 2;
Gc = calc_dual_graph(fmdl, lung_idx, struct('target_N', N_coarse, 'groups', grp));
Jf = calc_jacobian(mk_image(fmdl, sigma0));
Ac = Jf * Gc.EL;

% ---------------- forward model (finer, different mesh; as E4d) ----------------
ff = mk_library_model({'adult_male','boundary','left_lung','right_lung'}, [16 1 0.5], [0.05], maxh_fwd);
ff.stimulation = stim;  ff = mdl_normalize(ff, 0);
Nf = size(ff.elems,1);
lung_f = [ff.mat_idx{2}(:); ff.mat_idx{3}(:)];
sig0_f = ones(Nf,1);  sig0_f(lung_f) = sig_lung;
cf_all = interp_mesh(ff, 0);  vf_all = get_elem_volume(ff);
cf = cf_all(lung_f,:);  vf = vf_all(lung_f);
img_f = mk_image(ff, sig0_f);  vh = fwd_solve(img_f);  VH = vh.meas;

% ---------------- phantom geometry (E4d constants, fine mesh) ----------------
C = phantom_consts(cf, vf, sc, ell, T1, k_ins, Vt);
[u_f, dP_f, phi_f] = phantom_fields(cf, C);
[u_r, dP_r, phi_r] = phantom_fields(G.cent, C);
[~,   ~,    phi_c] = phantom_fields(Gc.cent, C);
selA_r = phi_r <= 0.5;  selA_c = phi_c <= 0.5;               % region S = anterior/A side
tt = (0:T-1) * dt;
vb = @(t) Vt * ((mod(t,Tb) < Ti) .* sin(pi/2*mod(t,Tb)/Ti) + ...
                (mod(t,Tb) >= Ti) .* cos(pi/2*(mod(t,Tb)-Ti)/(Tb-Ti)).^2);
vA = vb(tt);
fq = (0:T-1) / (T*dt);  fq(fq > fps/2) = fq(fq > fps/2) - fps;
q1 = zeros(1, T1);  kq = sc.kq;  q1(kq) = sin(pi*(kq-kq(1))/(kq(end)-kq(1))).^2;
qq = repmat(q1, 1, NB);
fprintf('E6: Q0max %.5f, delta_ref %.3f | lung nodes %d, clusters %d, fine lung elements %d\n', ...
        C.Q0max, C.delta_ref, G.N, Gc.N, numel(lung_f));

nC = numel(COND);
GT  = nan(nC, 3);                 % [opp, quadrature, lag ms] on true images
EST = nan(nC, numel(seeds), 8, numel(NLS));   % [HK raw, HK quad, GN opp, quadGN, FRIC, regout, Sang, GN lag] x noise
ORC = nan(nC, 3);                 % oracle FRIC, regout, Sang on true images
for ic = 1:nC
    cd_ = COND(ic);
    switch cd_.mech
        case 'none';  vP = vA;
        case 'delay'; vP = vb(tt - cd_.tau);
        case 'rc';    vP = real(ifft(fft(vA) ./ (1 + 1i*2*pi*fq*cd_.tau)));
    end
    Q0 = cd_.dose * C.Q0max;
    mk = @(u, phi, dP) (u .* (1 - phi)) * vA + (u .* phi) * vP + dP * (Q0 * qq);
    rho_f = mk(u_f, phi_f, dP_f);  rho_r = mk(u_r, phi_r, dP_r);
    % breaths from the true global volume (exactly periodic)
    Vg = G.vol' * rho_r;  [~, k0] = min(Vg(1:T1));
    kmin = k0 + (0:NB) * T1;  kmin = kmin(kmin <= T);  nb = numel(kmin) - 1;
    kei = arrayfun(@(b) kmin(b) - 1 + find(Vg(kmin(b):kmin(b+1)) == max(Vg(kmin(b):kmin(b+1))), 1), 1:nb);
    [Bt, lag_t] = pend_estimators(rho_r, G.vol, selA_r, kmin, kei, dt);
    GT(ic,:) = [mean(Bt(:,1:2), 1, 'omitnan'), lag_t];
    ORC(ic,:) = mean(Bt(:,3:5), 1, 'omitnan');
    if ic == 1; assert(GT(1,2) < 1e-6, 'E6 self-check: in-phase oracle quadGN must be 0'); end
    if strcmp(cd_.mech, 'delay') && Q0 == 0
        % hard A/P split of the soft partition (ell) mixes some vA into region P, so the regional lag is < tau
        assert(lag_t > 0.5e3*cd_.tau && lag_t < 1.2e3*cd_.tau, 'E6 self-check: true lag %.1f ms vs tau %.0f ms', lag_t, 1e3*cd_.tau);
    end
    % data (fine mesh), noise
    VI0 = zeros(numel(VH), T);
    for k = 1:T
        img_f.elem_data = sig0_f;  img_f.elem_data(lung_f) = sig_lung - rho_f(:,k);
        vv = fwd_solve(img_f);  VI0(:,k) = vv.meas;
    end
    for in = 1:numel(NLS)
    nl = NLS(in);
    for is = 1:numel(seeds)
        rs = RandStream('mt19937ar', 'Seed', seeds(is) + 100*ic + 1000*(in-1));
        VI = VI0 + nl * (norm(VI0 - VH, 'fro')/sqrt(numel(VI0))) * randn(rs, size(VI0));
        Yd = calc_difference_data(VH, VI, fmdl);
        rgn = -gn_frames(Ac, Yd, hp_w);
        gV  = Gc.vol' * rgn;
        TVg = arrayfun(@(b) gV(kei(b)) - gV(kmin(b)), (1:nb)');
        tiv = max(mean(rgn(:, kei) - rgn(:, kmin(1:nb)), 2), 0);
        o = struct('alpha', P1.alpha, 'delta', P1.dm*C.delta_ref, 'beta', P1.beta, 'gamma', 1e3, 'dt', dt, ...
                   'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, 'normalize', true, ...
                   'tau0', tau0, 'rho0', P1.kappa * sum(Gc.vol .* tiv) / sum(Gc.vol), 'vent_w', tiv);
        t0 = tic;
        [rho, m, ~, info] = hk_core(-Jf, speye(size(Jf,1)), Yd, Gc, o);
        QG = hk_pendelluft(Gc, m, selA_c, dt);
        F  = (diff(Gc.vol' * rho) / dt)';
        a  = (F' * QG) / (F' * F);
        Braw = 100 * insp_pv(QG, kmin, kei, dt) ./ TVg;
        Bq   = 100 * insp_pv(QG - a*F, kmin, kei, dt) ./ TVg;
        [Bg, lag_g] = pend_estimators(rgn, Gc.vol, selA_c, kmin, kei, dt);
        EST(ic, is, :, in) = [sum(mean(Braw, 1, 'omitnan')), sum(mean(Bq, 1, 'omitnan')), mean(Bg, 1, 'omitnan'), lag_g];
        fprintf(['%-26s n%.3f s%d | GT opp %5.2f q %5.2f lag %+6.1f | HK raw %5.2f q %5.2f | GN opp %5.2f qGN %5.2f ' ...
                 'FRIC %5.2f regout %5.2f Sang %5.2f lag %+6.1f | res %.0e %d it %.0f s\n'], cd_.name, nl, is, GT(ic,:), ...
                squeeze(EST(ic, is, :, in)), max(info.res), info.iters, toc(t0));
    end
    end
end
save(fullfile(here, ofile), 'COND', 'GT', 'EST', 'ORC', 'seeds', 'NLS', 'P1', 'sc', 'MODE');

if strcmp(MODE, 'noise')
    en = {'HK raw', 'HK quad', 'GN opp', 'GN quadGN', 'GN FRIC', 'GN regout', 'GN Sang'};  jj = [2 4 5 7];
    fprintf('\n============ E6 noise: mean (min..max over %d seeds), %%TV ============\n', numel(seeds));
    for in = 1:numel(NLS)
        fprintf('noise %.3f\n', NLS(in));
        for ic = 1:nC
            fprintf('  %-18s GTq %6.2f |', COND(ic).name, GT(ic,2));
            for j = jj; x = EST(ic,:,j,in); fprintf(' %s %6.2f (%5.2f..%5.2f) |', en{j}, mean(x), min(x), max(x)); end
            fprintf('\n');
        end
    end
    fprintf('\nHK vs GN quadrature: gain = mean / GT_q; margin = (mean - max M0 seed) / GT_q (> 0: separable from the in-phase floor)\n');
    for in = 1:numel(NLS)
        for j = [2 4]
            fprintf('  noise %.3f %-9s |', NLS(in), en{j});
            for ic = 2:nC
                x = EST(ic,:,j,in);
                fprintf(' %s gain %.2f margin %+.2f |', COND(ic).name, mean(x)/GT(ic,2), (mean(x) - max(EST(1,:,j,in)))/GT(ic,2));
            end
            fprintf(' floor %.2f\n', mean(EST(1,:,j,in)));
        end
    end
    fprintf('(the 1 %% level is in E6_results.mat / E6.log)\n');
    return
end

% ---------------- summary ----------------
M = squeeze(mean(EST, 2));
en = {'HK raw', 'HK quad', 'GN opp', 'GN quadGN', 'GN FRIC', 'GN regout', 'GN Sang'};
fprintf('\n============ E6: mean over seeds, %%TV (lag ms) ============\n');
fprintf('%-26s | %6s %6s %6s |', 'condition', 'GTopp', 'GTq', 'GTlag');  fprintf(' %9s', en{:});  fprintf(' %7s | oracle FRIC regout Sang\n', 'GN lag');
for ic = 1:nC
    fprintf('%-26s | %6.2f %6.2f %+6.1f |', COND(ic).name, GT(ic,:));  fprintf(' %9.2f', M(ic,1:7));
    fprintf(' %+7.1f | %6.2f %6.2f %6.2f\n', M(ic,8), ORC(ic,:));
end
pos = 2:nC;                                    % conditions with GT > 0
fprintf('\nTracking across mechanisms (M1-M4): ratio estimate / GT and Pearson r over the %d conditions\n', numel(pos));
fprintf('%-10s | %-22s | %-22s | %s\n', 'estimator', 'ratio to GT_q min..max', 'ratio to GT_opp min..max', 'r(GT_q)  r(GT_opp)  M0 false positive');
for j = 1:7
    rq = M(pos,j) ./ GT(pos,2);  ro = M(pos,j) ./ max(GT(pos,1), eps);
    c1 = corrcoef(M(pos,j), GT(pos,2));  c2 = corrcoef(M(pos,j), GT(pos,1));
    fprintf('%-10s | %8.2f .. %-8.2f    | %8.2f .. %-8.2f    | %+.2f    %+.2f     %.2f\n', en{j}, min(rq), max(rq), ...
            min([ro(GT(pos,1) > 0.05); NaN]), max([ro(GT(pos,1) > 0.05); NaN]), c1(1,2), c2(1,2), M(1,j));   % NaN if no GT_opp > 0
end
ia = ~strcmp({COND.mech}, 'none');
fprintf('GN lag / true lag (M2-M4): '); fprintf('%+.0f/%+.0f ', [M(ia,8)'; GT(ia,3)']); fprintf('ms\n');

% ======================================================================
function C = phantom_consts(cf, vf, sc, ell, T1, k_ins, Vt)
% as run_E4d_noncrime (constants from the FINE mesh)
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

function m = wmedian(x, w)
[xs, o] = sort(x);  cw = cumsum(w(o)) / sum(w);
m = xs(find(cw >= 0.5, 1));
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
