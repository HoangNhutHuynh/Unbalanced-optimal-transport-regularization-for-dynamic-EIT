%RUN_E4B_SIM  Experiment E4b: pendelluft DURING inspiration, known flux.
%
%   E4 (breath hold) cannot separate HK from image-based estimators: with a
%   constant total volume any regional swing IS pendelluft. E4b puts the
%   transfer inside a breath, where airway inflow and inter-regional
%   transfer co-occur (spontaneous-effort pendelluft, Yoshida et al. AJRCCM
%   2013): early in inspiration gas moves from the anterior (A) to the
%   posterior (P) region and returns later in inspiration.
%
%   Phantom (fine dual graph G of adult_male_16el_lungs):
%     soft regions: phi_P = sigmoid((n.(x - x_c) - s0)/ell), n TILTED, so the
%     boundary is not aligned with the voxel grid of the cluster graph Gc
%     rho(x,t) = v(t) u(x) + Q(t) (phi_P/|P| - phi_A/|A|)
%       v: tidal volume per unit volume (fast rise, sin), u = 1 + 0.3 phi_P
%       (normalized), Q: cumulative A->P transfer, Q0 sin^2 bump in early
%       inspiration (A->P then P->A). rho is a CHANGE w.r.t. end-expiration
%       and may be negative (A loses gas below its reference level).
%     GT PV = A->P volume across the hard interface phi_P = 0.5 during
%       inspiration = Q0 * gamma (gamma computed exactly).
%   Variants: 'overt'  - A's volume DECREASES early (Q0 = 3 Q0max)
%             'masked' - A's volume only slows down (Q0 = 0.8 Q0max), so an
%                        image-based "volume lost" estimator sees ~nothing.
%     Q0max = largest Q0 for which V_A(t) is still non-decreasing.
%             'none'   - Q0 = 0: in-phase but HETEROGENEOUS ventilation
%                        (u = 1 + 0.3 phi_P). SPECIFICITY control: any HK
%                        flux-PV here is a false positive (errors are then
%                        reported relative to the masked PV scale).
%   NOTE (identifiability): a masked transfer produces the same images as
%   a region that merely fills LATER without any inter-regional transfer;
%   no EIT method can separate the two. HK reports the minimal-action
%   (WFR) redistribution volume; 'none' tests that it stays ~0 when the
%   regions fill in phase.
%
%   Estimators of PV:
%     HK flux-PV : interface flux A->P during inspiration (hk_pendelluft)
%     FRIC-vol   : sum over elements of the volume LOST during inspiration
%                  (volume analogue of the FRIC idea, Adler et al. 2024)
%     region-out : volume lost by region A during inspiration (Sang et al.
%                  2020 style, regional out-of-phase volume)
%   Image-based estimators are applied to the GROUND-TRUTH images too: this
%   is their bias with perfect images (no reconstruction error).
%
%   HK grid: alpha x delta (= delta_ref*{1,2,4,8}) x rho0 (= kappa*mean TIV,
%   kappa in {0,3}) x ventilation mode {off,on}. The mode profile w is the
%   TIV map of a GN-cluster reconstruction (data-driven, NOT the truth).
%   Baselines (closed form, cluster graph, as E4): GN-cluster, L2-ST.
%
%   Needs EIDORS. Inverse crime remains in the FEM forward mesh (shared).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
if ~exist('mk_library_model','file')
    error('EIDORS not on path - run its startup.m first.');
end
rng(3);

% ---------------- knobs ----------------
variants  = {'none', 'masked', 'overt'};  % 'none' = SPECIFICITY control (Q0 = 0)
fps   = 10;   dt = 1/fps;
T1    = 31;                       % frames: 1 = end-expiration reference
k_ins = 1:16;                     % frames of inspiration (1 -> 16)
Vt    = 0.10;                     % tidal change per unit volume (|dsigma| units)
kq    = 2:10;                     % transfer window (A->P until 6, back until 10)
tilt  = [0.30 -1.00 0.25];        % boundary normal (posterior = -y, tilted)
ell   = 0.04;                     % transition width of the soft regions
nl    = 0.005;                    % additive noise level
sig_lung  = 0.4;
N_coarse  = 800;
tau0      = 100;
it_coarse = 4000;   it_refine = 20000;
alphas = 1e-3;   dmult = [2 4 8];   kappas = 3;   vents = [0 1];
beta   = 1e-2;          % E4: PV-selected beta
% Protocol configurations, reported for every variant WITHOUT oracle:
%   P0 (fixed after E4 / E4b-v2): no ventilation mode. E4b-v3 showed a FALSE
%      POSITIVE PV of ~0.0036 (41% of the masked scale) on 'none'.
%   P1 (chosen AFTER that result): ventilation mode on (w = TIV map), so
%      in-phase heterogeneous filling is free and only out-of-phase
%      redistribution costs transport. Needs validation on an independent
%      phantom (E4c) before it can be trusted.
% (E4b-v3: kappa in {2,3,5} changed PV by < 5% -> fixed to 3; alpha = 1e-2
%  degraded the images at delta >= 4x -> fixed to 1e-3)
proto  = struct('name', {'P0 (no vent)', 'P1 (vent, TIV)'}, 'alpha', {1e-3, 1e-3}, ...
                'dm', {4, 4}, 'kappa', {3, 3}, 'vent', {0, 1});

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

% ---------------- soft regions (G node order!) ----------------
n   = tilt / norm(tilt);
xc  = sum(G.cent .* G.vol, 1) / sum(G.vol);
s   = (G.cent - xc) * n';
s0  = wmedian(s, G.vol);
phiP = 1 ./ (1 + exp(-(s - s0)/ell));          % posterior membership
phiA = 1 - phiP;
VP = sum(G.vol .* phiP);   VA = sum(G.vol .* phiA);
hardP = phiP > 0.5;                             % hard interface for PV
sc  = (Gc.cent - xc) * n';
hardP_c = 1 ./ (1 + exp(-(sc - s0)/ell)) > 0.5; % same interface on clusters
mixed = accumarray(Gc.cluster, hardP, [], @(x) any(x) && ~all(x));
fprintf('soft regions: |A| %.4f |P| %.4f | clusters straddling the interface: %d of %d\n', ...
        VA, VP, nnz(mixed), Gc.N);
dP = phiP/VP - phiA/VA;                         % unit transfer pattern (vol'*dP = 0)
gamma = sum(G.vol(hardP) .* dP(hardP));         % hard-P gain per unit Q

% ---------------- dynamics ----------------
v = zeros(1,T1);
v(k_ins) = Vt * sin(pi/2 * (k_ins-1)/(numel(k_ins)-1));         % fast early flow
k_exp = k_ins(end):T1;
v(k_exp) = Vt * cos(pi/2 * (k_exp-k_ins(end))/(T1-k_ins(end))).^2;
u = 1 + 0.3*phiP;   u = u * sum(G.vol) / sum(G.vol .* u);       % vol-mean 1
qsh = zeros(1,T1);  qsh(kq) = sin(pi*(kq-kq(1))/(kq(end)-kq(1))).^2;  % Q/Q0
% Q0max: V_A(t) non-decreasing on the inspiration with transfer
aA  = sum(G.vol(~hardP) .* u(~hardP));          % A volume per unit v
gA  = -sum(G.vol(~hardP) .* dP(~hardP));        % A loss per unit Q (> 0)
dv  = diff(v(k_ins));  dq = diff(qsh(k_ins));
Q0max = min(aA*dv(dq > 0) ./ (gA*dq(dq > 0)));
fprintf('Q0max (A volume still non-decreasing) = %.5f\n', Q0max);

% ---------------- linear operators for baselines/profile ----------------
imdl = eidors_obj('inv_model', 'E4b');
imdl.fwd_model = fmdl;  imdl.reconst_type = 'difference';
imdl.jacobian_bkgnd.value = sigma0;
imdl.RtR_prior = @prior_noser;  imdl.hyperparameter.value = 1e-2;
Jf = calc_jacobian(calc_jacobian_bkgnd(imdl));
Ac = Jf * Gc.EL;
ins_steps = (1:T1-1)' < k_ins(end);              % steps 1..15
delta_ref = norm(sum(G.cent(hardP,:).*G.vol(hardP),1)/sum(G.vol(hardP)) - ...
                 sum(G.cent(~hardP,:).*G.vol(~hardP),1)/sum(G.vol(~hardP))) / pi;
fprintf('delta_ref = L_AP/pi = %.3f\n', delta_ref);

R = struct();
for iv = 1:numel(variants)
    vname = variants{iv};
    switch vname
        case 'overt';  Q0 = 3.0 * Q0max;
        case 'masked'; Q0 = 0.8 * Q0max;
        case 'none';   Q0 = 0;       % in-phase, heterogeneous ventilation only
    end
    rho_true = u * v + dP * (Q0 * qsh);          % N x T1 (change, may be < 0)
    ds_true  = -(G.EL * rho_true);
    PV_gt    = Q0 * gamma;
    VA_t     = G.vol(~hardP)' * rho_true(~hardP,:);
    fprintf('\n################ E4b-%s: Q0 = %.5f, PV_gt = %.5f, min rho = %.4f, A volume dips: %d\n', ...
            vname, Q0, PV_gt, min(rho_true(:)), any(diff(VA_t(k_ins)) < 0));

    % forward simulation
    img = mk_image(fmdl, sigma0);
    vh  = fwd_solve(img);  VH = vh.meas;  VI = zeros(numel(VH), T1);
    for k = 1:T1
        img.elem_data = sigma0 + ds_true(:,k);
        vv = fwd_solve(img);  VI(:,k) = vv.meas;
    end
    VI = VI + nl * (norm(VI - VH, 'fro')/sqrt(numel(VI))) * randn(size(VI));
    Yd = calc_difference_data(VH, VI, fmdl);
    rel_lung = @(DS) norm(DS(lung_idx,:) - ds_true(lung_idx,:), 'fro') / ...
                     norm(ds_true(lung_idx,:), 'fro');
    % relative PV error; for 'none' (PV_gt = 0) relative to the masked scale
    PV_scale = max(PV_gt, 0.8 * Q0max * gamma);
    pv_err = @(pv) abs(pv - PV_gt) / PV_scale;
    est    = @(DS) posthoc(DS, G, hardP, ins_steps);    % [FRIC-vol, region-out]

    % estimator bias on PERFECT images
    pe_gt = est(ds_true);
    fprintf('image-based estimators on the GT images: FRIC-vol %.5f, region-out %.5f (GT PV %.5f)\n', ...
            pe_gt(1), pe_gt(2), PV_gt);
    res = {'(GT images) FRIC-vol', 0, pe_gt(1); '(GT images) region-out', 0, pe_gt(2)};

    % baselines
    hps = logspace(-4, 1, 11);  tb = zeros(numel(hps), 4);
    for h = 1:numel(hps)
        x = Gc.EL * gn_frames(Ac, Yd, hps(h));
        tb(h,:) = [hps(h), rel_lung(x), est(x)];
    end
    [~, je] = min(tb(:,2));
    res(end+1,:) = {'GN-cluster (sel err) FRIC-vol',   tb(je,2), tb(je,3)}; %#ok<SAGROW>
    res(end+1,:) = {'GN-cluster (sel err) region-out', tb(je,2), tb(je,4)}; %#ok<SAGROW>
    [~, jp] = min(pv_err(tb(:,4)));
    res(end+1,:) = {'GN-cluster (sel PV) region-out',  tb(jp,2), tb(jp,4)}; %#ok<SAGROW>
    hp_gn = tb(je,1);
    lss = logspace(-5, 1, 7);  lts = logspace(-4, 2, 7);  tl = [];
    for ls = lss
        for lt = lts
            x = Gc.EL * l2st(Ac, Yd, ls, lt);
            tl(end+1,:) = [ls, lt, rel_lung(x), est(x)]; %#ok<AGROW>
        end
    end
    [~, je] = min(tl(:,3));
    res(end+1,:) = {'L2-ST (sel err) FRIC-vol',   tl(je,3), tl(je,4)}; %#ok<SAGROW>
    res(end+1,:) = {'L2-ST (sel err) region-out', tl(je,3), tl(je,5)}; %#ok<SAGROW>
    [~, jp] = min(pv_err(tl(:,5)));
    res(end+1,:) = {'L2-ST (sel PV) region-out',  tl(jp,3), tl(jp,5)}; %#ok<SAGROW>

    % data-driven ventilation profile and FRC level from GN-cluster
    rho_gn = -gn_frames(Ac, Yd, hp_gn);                     % Nc x T1
    tiv    = max(rho_gn(:, k_ins(end)) - rho_gn(:, 1), 0);  % TIV density map
    tiv_m  = sum(Gc.vol .* tiv) / sum(Gc.vol);

    % HK sweep
    o0 = struct('gamma',1e3,'dt',dt,'max_iter',it_coarse,'tol',1e-4,'verbose',0, ...
                'use_eta',false,'normalize',true,'tau0',tau0,'beta',beta);
    gl = [];  img_c = {};
    for alpha = alphas
      for dm = dmult
        for kap = kappas
          for ven = vents
            o = o0;  o.alpha = alpha;  o.delta = dm*delta_ref;  o.rho0 = kap*tiv_m;
            if ven; o.vent_w = tiv; end
            [rho, m, ~, info] = hk_core(-Jf, speye(size(Jf,1)), Yd, Gc, o);
            DS  = -(Gc.EL * rho);
            [~, pvf] = hk_pendelluft(Gc, m, hardP_c, dt, ins_steps);
            pe  = est(DS);
            gl(end+1,:) = [alpha dm kap ven rel_lung(DS) pvf pe max(info.res)]; %#ok<AGROW>
            fprintf('HK %s: a=%.0e d=%gx k=%g vent=%d | err %.4f | flux-PV %.5f | FRIC %.5f reg %.5f | res %.1e\n', ...
                    vname, alpha, dm, kap, ven, gl(end,5), pvf, pe(1), pe(2), max(info.res));
          end
        end
      end
    end
    % refine the 2 best by flux-PV error
    [~, op] = sort(pv_err(gl(:,6)));
    best = struct('pvf', Inf);
    for r = op(1:2)'
        o = o0;  o.alpha = gl(r,1);  o.delta = gl(r,2)*delta_ref;  o.rho0 = gl(r,3)*tiv_m;
        if gl(r,4); o.vent_w = tiv; end
        o.max_iter = it_refine;  o.tol = 1e-6;
        [rho, m, ~, info] = hk_core(-Jf, speye(size(Jf,1)), Yd, Gc, o);
        DS = -(Gc.EL * rho);
        [QG, pvf] = hk_pendelluft(Gc, m, hardP_c, dt, ins_steps);
        er = rel_lung(DS);
        fprintf('HK refine: a=%.0e d=%gx k=%g vent=%d -> err %.4f (coarse %.4f) | flux-PV %.5f (coarse %.5f) | res %.1e\n', ...
                gl(r,1), gl(r,2), gl(r,3), gl(r,4), er, gl(r,5), pvf, gl(r,6), max(info.res));
        if pv_err(pvf) < pv_err(best.pvf)
            best = struct('err', er, 'pvf', pvf, 'row', r, 'o', o, 'rho', rho, 'm', m, ...
                          'info', info, 'QG', QG);
        end
    end
    res(end+1,:) = {sprintf('HK (sel PV) flux-PV [a=%.0e d=%gx k=%g v=%d]', ...
                    best.o.alpha, gl(best.row,2), gl(best.row,3), gl(best.row,4)), best.err, best.pvf}; %#ok<SAGROW>
    % pre-specified protocol configuration (no oracle)
    for ip = 1:numel(proto)
        pp = proto(ip);
        pr = find(gl(:,1) == pp.alpha & gl(:,2) == pp.dm & gl(:,3) == pp.kappa & ...
                  gl(:,4) == pp.vent, 1);
        if ~isempty(pr)
            res(end+1,:) = {sprintf('HK %s flux-PV (no oracle)', pp.name), gl(pr,5), gl(pr,6)}; %#ok<SAGROW>
        end
    end
    % honest robustness: HK flux-PV over the WHOLE grid (no oracle)
    fprintf('HK flux-PV over all %d configs: median %.5f, IQR [%.5f %.5f] (GT %.5f)\n', ...
            size(gl,1), median(gl(:,6)), prctile_(gl(:,6), 25), prctile_(gl(:,6), 75), PV_gt);
    for kap = kappas
        for ven = vents
            sel = gl(:,3) == kap & gl(:,4) == ven;
            fprintf('   kappa=%g vent=%d: flux-PV median %.5f, best |err| %.3f\n', ...
                    kap, ven, median(gl(sel,6)), min(pv_err(gl(sel,6))));
        end
    end

    % decomposition of the selected HK run
    b  = best;  zt = b.info.zeta;  ins = ins_steps';
    gV = dt * (Gc.vol' * zt);
    fprintf('HK decomposition: global source over inspiration %.5f (GT %.5f)', ...
            sum(gV(ins)), (v(k_ins(end)) - v(1)) * sum(G.vol .* u));
    if ~isempty(b.info.vent_c)
        fprintf(' | of which ventilation mode %.5f', dt * sum(b.info.vent_c(ins)));
    end
    fprintf('\n');

    % report
    fprintf('\n============ E4b-%s RESULTS (GT PV = %.5f; PV error scale %.5f) ============\n', ...
            vname, PV_gt, PV_scale);
    fprintf('%-58s %8s %9s %10s\n', 'Method / estimator', 'rel err', 'PV', '|PV-GT|/sc');
    for r = 1:size(res,1)
        fprintf('%-58s %8.4f %9.5f %10.3f\n', res{r,1}, res{r,2}, res{r,3}, pv_err(res{r,3}));
    end
    R.(vname) = struct('res', {res}, 'gl', gl, 'tb', tb, 'tl', tl, 'best', best, ...
                     'PV_gt', PV_gt, 'Q0', Q0, 'pe_gt', pe_gt);
end
save E4b_results.mat R variants delta_ref Q0max gamma

% ======================================================================
function pe = posthoc(DS, G, hardP, ins_steps)
% image-based PV estimators from element dsigma (Ne x T1); rho = -dsigma
rho = -(G.EL' * DS);                            % fine lung graph, N x T1
d   = diff(rho, 1, 2);  d = d(:, ins_steps);     % per-step changes in inspiration
fric = sum(sum(G.vol .* max(-d, 0)));           % volume lost by any element
dVA  = G.vol(~hardP)' * d(~hardP, :);           % region A volume change per step
reg  = sum(max(-dVA, 0));                       % volume lost by region A
pe = [fric, reg];
end

function m = wmedian(x, w)
[xs, o] = sort(x);  cw = cumsum(w(o)) / sum(w);
m = xs(find(cw >= 0.5, 1));
end

function p = prctile_(x, q)
% percentile without the Statistics Toolbox (linear interpolation)
x = sort(x(:));  n = numel(x);
r = 1 + (n-1)*q/100;  lo = floor(r);  hi = ceil(r);
p = x(lo) + (r - lo) * (x(hi) - x(lo));
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
