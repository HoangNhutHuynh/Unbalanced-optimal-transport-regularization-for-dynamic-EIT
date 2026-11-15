%RUN_E4_SIM  Experiment E4: simulated pendelluft phantom with ground truth.
%
%   Phantom on mk_library_model('adult_male_16el_lungs'):
%     frames 1..T1, dt = 1/fps:
%       phase A (inspiration): global ventilation v(k) ramps 0 -> Vt
%       phase B (breath hold) : v constant, pure pendelluft p(k) moves gas
%                               anterior <-> posterior with ZERO net volume
%     rho_true = v(k)*1_lung + p(k)*w,  w = +/-1 (ant/post), vol-balanced.
%     Delta sigma = -rho_true on lung elements.
%
%   Ground truth pendelluft volume PV_gt = swing of posterior gas content
%   during the hold (all of it must cross the ant/post interface).
%
%   Reconstructions (v6): HK-dEIT (sweep over alpha, delta = (L_ap/pi)*[.5 1 2 4],
%   beta; refine of the best by rel err and by PV error), and closed-form
%   baselines GN-cluster, L2-ST-cluster (= random-walk Kalman smoother MAP;
%   HK with the action replaced by an L2 time penalty) and GN-element.
%   EIDORS' temporal-prior / Kalman / GN solvers are not used: they form
%   dense Ne x Ne matrices (> 5 GB at Ne = 26k). Every method is reported
%   with hyperparameters chosen by the ground-truth oracle on rel err AND on
%   PV error. PV: HK flux-based and post-hoc (posterior swing); baselines
%   post-hoc only.
%
%   Graphs: the phantom lives on the FINE dual graph G (one node per lung
%   element); HK reconstructs on a voxel-CLUSTER graph Gc (~N_coarse nodes,
%   calc_dual_graph v2), which partly avoids the inverse crime and makes CP
%   converge in reasonable time. The FEM forward mesh is still shared -
%   redo with a finer forward mesh before using numbers in the paper.
%
%   Requires EIDORS on the path, e.g.:
%   run('E:\Hoang\EIT\prior_functional\eidors-v3.12-ng\eidors\startup.m')
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars -except E4_SETUP_ONLY; clc;
if ~exist('mk_library_model','file')
    error(['EIDORS not on path. Run its startup.m first, e.g. ', ...
           'run(''E:\\Hoang\\EIT\\prior_functional\\eidors-v3.12-ng\\eidors\\startup.m'')']);
end
rng(2);

% ---------------- knobs ----------------
fps   = 10;    dt = 1/fps;
T1    = 26;                       % frames incl. baseline frame 1
kinsp = 2:12;                     % inspiration ramp
khold = 13:T1;                    % breath hold with pendelluft
Vt    = 0.10;                     % global ventilation amplitude (in |dsigma|)
Pamp  = 0.03;                     % pendelluft amplitude
nl    = 0.005;                    % additive noise level (0.5%)
sig_lung = 0.4;                   % baseline lung conductivity (bg = 1)
tau0      = 100;                  % initial CP step (hk_core v5 then adapts it)
N_coarse  = 800;                  % voxel clusters of the reconstruction graph
it_coarse = 4000;                 % coarse sweep: 48 configs x ~40 s
it_refine = 20000;                % refine: <= 4 configs x ~3.5 min

% ---------------- model ----------------
fmdl = mk_library_model('adult_male_16el_lungs');
fmdl.stimulation = mk_stim_patterns(16,1,[0,1],[0,1],{'no_meas_current'},1);
fmdl = mdl_normalize(fmdl, 0);
Ne = size(fmdl.elems,1);
if numel(fmdl.mat_idx) < 3
    error('expected mat_idx{2},{3} = lungs in library model');
end
lung_idx = [fmdl.mat_idx{2}(:); fmdl.mat_idx{3}(:)];

sigma0 = ones(Ne,1);  sigma0(lung_idx) = sig_lung;

% dual graph + anterior/posterior split (by centroid, 2nd coordinate)
% NOTE (v6 bug fix): post must follow the NODE ORDER of G, i.e.
% G.node_idx = unique(lung_idx) (sorted) - not the order of lung_idx.
G    = calc_dual_graph(fmdl, lung_idx);
cent = interp_mesh(fmdl, 0);
yl   = cent(G.node_idx, 2);
post = yl < median(yl);           % "dependent" half (prep_E1 checks the sign of +y)
fprintf('lung_idx sorted: %d  (if 0, runs before v6 used a SCRAMBLED ant/post split)\n', ...
        issorted(lung_idx));
w    = double(~post) - double(post);          % +1 ant, -1 post
w    = w - (G.vol'*w)/sum(G.vol);             % volume-balanced: vol'*w = 0

% reconstruction graph: voxel clusters, never mixing left/right lung
grp = zeros(Ne,1);  grp(fmdl.mat_idx{2}) = 1;  grp(fmdl.mat_idx{3}) = 2;
Gc  = calc_dual_graph(fmdl, lung_idx, struct('target_N', N_coarse, 'groups', grp));
post_c = Gc.cent(:,2) < median(yl);           % same ant/post threshold
fprintf('fine graph: N = %d, E = %d | cluster graph: N = %d, E = %d (cube %.3g)\n', ...
        G.N, G.E, Gc.N, Gc.E, Gc.h_cluster);

% ---------------- ground-truth dynamics ----------------
v = zeros(1,T1);  p = zeros(1,T1);
v(kinsp) = Vt * sin(linspace(0, pi/2, numel(kinsp))).^2;
v(khold) = Vt;
p(khold) = Pamp * sin(2*pi*(0:numel(khold)-1)/numel(khold));
rho_true = max(v + w*p, 0);                   % N x T1 (>=0 by construction)
assert(min(v + w*p, [], 'all') > -1e-12, 'phantom went negative - reduce Pamp');

mass_post   = G.vol(post)' * rho_true(post,:);
PV_gt       = max(mass_post(khold)) - min(mass_post(khold));
fprintf('GT pendelluft volume PV_gt = %.5f (impedance-equivalent units)\n', PV_gt);

% ---------------- forward simulation ----------------
img = mk_image(fmdl, sigma0);
vh  = fwd_solve(img);  VH = vh.meas;
M   = numel(VH);  VI = zeros(M, T1);
for k = 1:T1
    ds = -(G.EL * rho_true(:,k));          % Ne x 1, nonzero on lungs only
    img.elem_data = sigma0 + ds;
    vv = fwd_solve(img);  VI(:,k) = vv.meas;
end
VI = VI + nl * (norm(VI - VH, 'fro')/sqrt(numel(VI))) * randn(M, T1);

% ---------------- inverse model ----------------
imdl = eidors_obj('inv_model', 'E4-HK');
imdl.fwd_model      = fmdl;
imdl.reconst_type   = 'difference';
imdl.jacobian_bkgnd.value = sigma0;
imdl.RtR_prior      = @prior_noser;        % used only by baselines
imdl.hyperparameter.value = 1e-2;

ds_true = -(G.EL * rho_true);              % Ne x T1 ground-truth dsigma
rel_lung = @(DS) norm(DS(lung_idx,:) - ds_true(lung_idx,:), 'fro') ...
               / norm(ds_true(lung_idx,:), 'fro');

% representation floor of the cluster graph: volume-weighted cluster
% average of the ground truth = best cluster-wise constant approximation
ve = zeros(Ne,1);  ve(G.node_idx) = G.vol;
rho_c_true = (Gc.EL' * (ve .* (-ds_true))) ./ Gc.vol;
err_floor  = rel_lung(-(Gc.EL * rho_c_true));
fprintf('cluster-graph representation floor: rel err (lung) = %.4f\n', err_floor);

diam = max(max(fmdl.nodes) - min(fmdl.nodes));
save E4_setup.mat fmdl imdl VH VI G Gc lung_idx post post_c sigma0 rho_true ...
     ds_true PV_gt dt T1 kinsp khold v diam
fprintf('saved E4_setup.mat (used by diag_cp_E4.m)\n');
if exist('E4_SETUP_ONLY','var') && E4_SETUP_ONLY
    return                         % called from diag_cp_E4: setup only
end

results = {};

% ---------------- HK-dEIT sweep ----------------
% (P4) transport is cheap below a distance ~ pi*delta, so delta is set from
% the geometry of the phantom: L_ap = distance between the volume-weighted
% centroids of the anterior and posterior lung halves; delta_ref = L_ap/pi.
c_post = sum(G.cent(post,:)  .* G.vol(post),  1) / sum(G.vol(post));
c_ant  = sum(G.cent(~post,:) .* G.vol(~post), 1) / sum(G.vol(~post));
L_ap   = norm(c_ant - c_post);
delta_ref = L_ap / pi;
fprintf('ant/post centroid distance L_ap = %.3f -> delta_ref = L_ap/pi = %.3f\n', L_ap, delta_ref);

o0 = struct('gamma',1e3,'dt',dt,'lung',lung_idx,'max_iter',it_coarse, ...
            'tol',1e-4,'verbose',0,'use_eta',false,'normalize',true, ...
            'tau0',tau0, 'G',Gc);   % hk_core v5: tau adapts from tau0
alphas = [1e-5 1e-4 1e-3 1e-2];
deltas = delta_ref * [0.5 1 2 4];
betas  = [1e-3 1e-2 1e-1];
imdl.solve = @inv_solve_diff_hk;
hmask = ismember(1:T1-1, khold(1:end-1))';     % steps inside the hold (for PV)
pv_err = @(pv) abs(pv - PV_gt) / PV_gt;
% gl columns: alpha delta beta relerr maxres PV_flux PV_posthoc
gl = [];
for alpha = alphas
  for delta = deltas
    for beta = betas
        o = o0;  o.alpha = alpha;  o.delta = delta;  o.beta = beta;
        imdl.inv_solve_diff_hk = o;
        img_hk = inv_solve(imdl, VH, VI);
        er  = rel_lung(img_hk.elem_data);
        rs_ = img_hk.hk.info.res;
        [~, pvf] = hk_pendelluft(Gc, img_hk.hk.flux, post_c, dt, hmask);
        pvp = posthoc_pv(img_hk.elem_data, G, post, khold);
        gl(end+1,:) = [alpha delta beta er max(rs_) pvf pvp]; %#ok<SAGROW>
        fprintf(['HK coarse: a=%.0e d=%.3g b=%.0e | err %.4f | PV flux %.5f posthoc %.5f | ' ...
                 'rP %.1e rD %.1e tau %.2g\n'], alpha, delta, beta, er, pvf, pvp, ...
                 rs_(1), rs_(2), img_hk.hk.info.tau);
    end
  end
end

% refine the 2 best by rel err AND the 2 best by PV error (flux-based)
[~, oe] = sort(gl(:,4));
[~, op] = sort(pv_err(gl(:,6)));
cand = unique([oe(1:2); op(1:2)], 'stable');
ref = struct('row', {}, 'err', {}, 'pvf', {}, 'pvp', {}, 'img', {}, 'o', {});
for r = cand(:)'
    o = o0;  o.alpha = gl(r,1);  o.delta = gl(r,2);  o.beta = gl(r,3);
    o.max_iter = it_refine;  o.tol = 1e-6;  o.verbose = 5000;
    imdl.inv_solve_diff_hk = o;
    fprintf('HK refine: a=%.0e d=%.3g b=%.0e ...\n', o.alpha, o.delta, o.beta);
    img_hk = inv_solve(imdl, VH, VI);
    er  = rel_lung(img_hk.elem_data);
    [~, pvf] = hk_pendelluft(Gc, img_hk.hk.flux, post_c, dt, hmask);
    pvp = posthoc_pv(img_hk.elem_data, G, post, khold);
    % task-level convergence: same config at it_coarse vs it_refine
    fprintf(['   -> err %.4f (coarse %.4f) | PV flux %.5f (coarse %.5f) | posthoc %.5f (coarse %.5f)' ...
             ' | %d it, rP %.1e rD %.1e, %d tau changes\n'], er, gl(r,4), pvf, gl(r,6), ...
             pvp, gl(r,7), img_hk.hk.info.iters, img_hk.hk.info.res(1), ...
             img_hk.hk.info.res(2), img_hk.hk.info.nadapt);
    ref(end+1) = struct('row', r, 'err', er, 'pvf', pvf, 'pvp', pvp, 'img', img_hk, 'o', o); %#ok<SAGROW>
end
[~, ie] = min([ref.err]);            hk_e = ref(ie);     % selected by rel err
[~, ip] = min(pv_err([ref.pvf]));    hk_p = ref(ip);     % selected by PV error
for s = {hk_e, hk_p}
    b = s{1};
    if b.o.alpha == min(alphas) || b.o.alpha == max(alphas)
        warning('E4: selected alpha %.0e on the edge of the grid.', b.o.alpha);
    end
    if b.o.delta == min(deltas) || b.o.delta == max(deltas)
        warning('E4: selected delta %.3g on the edge of the grid.', b.o.delta);
    end
    if b.o.beta == min(betas) || b.o.beta == max(betas)
        warning('E4: selected beta %.0e on the edge of the grid.', b.o.beta);
    end
end
results(end+1,:) = {'HK (sel. rel err) flux-PV',    hk_e.err, hk_e.pvf};
results(end+1,:) = {'HK (sel. rel err) posthoc-PV', hk_e.err, hk_e.pvp};
results(end+1,:) = {'HK (sel. PV)      flux-PV',    hk_p.err, hk_p.pvf};
results(end+1,:) = {'HK (sel. PV)      posthoc-PV', hk_p.err, hk_p.pvp};

% ---- transport/source decomposition checks (claims C3, C5, C7) ----
% step k goes from frame k to frame k+1
st_ins = 1:(kinsp(end)-1);                     % steps over the inspiration ramp
st_hld = kinsp(end):(T1-1);                    % steps inside the breath hold
Vgt_ins = (v(kinsp(end)) - v(1)) * sum(G.vol);
dec = struct();
for s = {{'sel. rel err', hk_e}, {'sel. PV', hk_p}}
    nm = s{1}{1};  b = s{1}{2};
    src  = b.img.hk.source;                                % N x T
    gsrc = dt * (Gc.vol' * src);                           % 1 x T global source
    psrc = dt * ((Gc.vol .* post_c)' * src);               % 1 x T posterior source
    fprintf('\n--- HK decomposition (%s: a=%.0e d=%.3g b=%.0e) ---\n', nm, b.o.alpha, b.o.delta, b.o.beta);
    fprintf('global source, inspiration : %+.5f   (GT %+.5f)\n', sum(gsrc(st_ins)), Vgt_ins);
    fprintf('global source, hold        : %+.5f   (GT 0: volume constant)\n', sum(gsrc(st_hld)));
    fprintf('posterior |source|, hold   :  %.5f   (GT 0; compare PV_gt = %.5f)\n', ...
            sum(abs(psrc(st_hld))), PV_gt);
    fprintf('PV flux / PV posthoc       :  %.5f / %.5f  (transport share of the swing %.2f)\n', ...
            b.pvf, b.pvp, b.pvf / max(b.pvp, eps));
    dec.(matlab.lang.makeValidName(nm)) = struct('gsrc', gsrc, 'psrc', psrc);
end

% ---------------- baselines: oracle hp sweep, both selection rules ------
% EIDORS' own temporal-prior, Kalman and GN solvers form dense Ne x Ne
% matrices (Ne = 26k here: > 5 GB each), so the baselines are solved
% EXACTLY by closed forms - on the SAME cluster graph as HK (fair
% parameterization) plus GN on the full element mesh:
%   GN-cluster : per-frame Tikhonov, NOSER prior diag(A'A)^0.5 (Wiener form)
%   L2-ST      : min sum||y-A x||^2 + ls x'Rx + lt sum||x_{t+1}-x_t||^2
%                (R = NOSER); this is the MAP of a random-walk Kalman
%                SMOOTHER and equals HK with the transport action replaced
%                by an L2 time penalty and without rho >= 0 (clean ablation)
%   GN-element : per-frame Tikhonov/NOSER on all Ne elements (Wiener form);
%                not limited by the cluster representation floor
Jf = calc_jacobian(calc_jacobian_bkgnd(imdl));        % M x Ne,  y ~ Jf*dsigma
Yd = calc_difference_data(VH, VI, fmdl);              % M x T1
Ac = Jf * Gc.EL;                                      % M x Nc (cluster values)
bl = {};
hps = logspace(-4, 1, 11);
% GN on clusters and on elements
for b = 1:2
    if b == 1; Ab = Ac; map = Gc.EL; nmb = 'GN-cluster'; else; Ab = Jf; map = []; nmb = 'GN-element'; end
    tab = zeros(numel(hps), 3);  X = cell(1, numel(hps));
    for h = 1:numel(hps)
        x = gn_frames(Ab, Yd, hps(h));
        if ~isempty(map); x = map * x; end
        tab(h,:) = [hps(h), rel_lung(x), posthoc_pv(x, G, post, khold)];
        X{h} = x;
    end
    [~, je] = min(tab(:,2));  [~, jp] = min(pv_err(tab(:,3)));
    results(end+1,:) = {sprintf('%s (sel. rel err)', nmb), tab(je,2), tab(je,3)}; %#ok<SAGROW>
    results(end+1,:) = {sprintf('%s (sel. PV)', nmb),      tab(jp,2), tab(jp,3)}; %#ok<SAGROW>
    bl(end+1,:) = {nmb, tab, X([je jp])}; %#ok<SAGROW>
    fprintf('%s: best rel err %.4f at hp %.0e | best PV %.5f at hp %.0e\n', ...
            nmb, tab(je,2), tab(je,1), tab(jp,3), tab(jp,1));
end
% L2 space-time on clusters
lss = logspace(-5, 1, 7);  lts = logspace(-4, 2, 7);
tab = zeros(numel(lss)*numel(lts), 4);  X = cell(1, size(tab,1));  q = 0;
for ls = lss
    for lt = lts
        q = q + 1;
        x = Gc.EL * l2st(Ac, Yd, ls, lt);
        tab(q,:) = [ls, lt, rel_lung(x), posthoc_pv(x, G, post, khold)];
        X{q} = x;
    end
end
[~, je] = min(tab(:,3));  [~, jp] = min(pv_err(tab(:,4)));
results(end+1,:) = {'L2-ST cluster (sel. rel err)', tab(je,3), tab(je,4)};
results(end+1,:) = {'L2-ST cluster (sel. PV)',      tab(jp,3), tab(jp,4)};
bl(end+1,:) = {'L2-ST cluster', tab, X([je jp])};
fprintf('L2-ST: best rel err %.4f at ls %.0e lt %.0e | best PV %.5f at ls %.0e lt %.0e\n', ...
        tab(je,3), tab(je,1), tab(je,2), tab(jp,4), tab(jp,1), tab(jp,2));
for jj = [je jp]
    if any(tab(jj,1) == lss([1 end])) || any(tab(jj,2) == lts([1 end]))
        warning('E4: L2-ST selection on the edge of its (ls, lt) grid.');
    end
end

% ---------------- report ----------------
fprintf('\n================== E4 RESULTS ==================\n');
fprintf('GT pendelluft volume PV_gt = %.5f\n', PV_gt);
fprintf('cluster-graph representation floor (HK cannot beat it): rel err %.4f\n\n', err_floor);
fprintf('%-32s  %-10s  %-10s  %-10s\n', 'Method', 'rel err', 'PV', '|PV-GT|/GT');
for r = 1:size(results,1)
    fprintf('%-32s  %-10.4f  %-10.5f  %-10.3f\n', results{r,:}, pv_err(results{r,3}));
end
fprintf(['\nflux-PV = HK interface flux (reconstruction-integrated); posthoc-PV =\n' ...
         'swing of the reconstructed posterior gas content (what an image-based\n' ...
         'method can measure). "sel." = hyperparameters chosen by the oracle on\n' ...
         'rel err or on PV error (both use the ground truth, for every method).\n']);

save E4_results.mat results hk_e hk_p ref gl bl dec PV_gt rho_true ds_true G Gc ...
     lung_idx post post_c khold dt err_floor L_ap delta_ref

% ======================================================================
function pv = posthoc_pv(elem_data, G, post, khold)
% post-hoc pendelluft estimate from any reconstruction: swing of the
% posterior gas content (-dsigma, volume-weighted) during the hold
rho_rec = -(G.EL' * elem_data);                 % N x T1 (fine lung graph)
mp = G.vol(post)' * rho_rec(post, :);
pv = max(mp(khold)) - min(mp(khold));
end

function X = gn_frames(A, Y, hp)
% per-frame Tikhonov with the NOSER prior R = diag(diag(A'A))^0.5 in the
% Wiener form  X = P A' (A P A' + hp^2 I)^-1 Y,  P = R^-1  (M x M solve)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end

function X = l2st(A, Y, ls, lt)
% exact minimizer of sum_t ||y_t - A x_t||^2 + ls x_t' R x_t
%                   + lt sum_t ||x_{t+1} - x_t||^2,   R = NOSER (as gn_frames)
% The temporal path Laplacian (Neumann) is diagonalized by the DCT-II basis
% V (eigenvalues 2 - 2cos(pi k/T1)), so the space-time system splits into
% T1 independent n x n solves.
[n, T1] = deal(size(A,2), size(Y,2));
H   = A' * A;
Rs  = diag(sqrt(diag(H)));
V   = cos(pi * ((0:T1-1)' + 0.5) * (0:T1-1) / T1);     % T1 x T1, columns = modes
V   = V ./ sqrt(sum(V.^2, 1));
lam = 2 - 2*cos(pi * (0:T1-1) / T1);
B   = (A' * Y) * V;                                    % n x T1 in mode space
Z   = zeros(n, T1);
for k = 1:T1
    Rk = chol(H + ls*Rs + (lt*lam(k))*eye(n));
    Z(:,k) = Rk \ (Rk' \ B(:,k));
end
X = Z * V';
end
