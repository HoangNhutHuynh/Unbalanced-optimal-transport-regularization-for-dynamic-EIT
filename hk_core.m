function [rho, m, eta, info] = hk_core(A, W, Y, G, o)
%HK_CORE  Preconditioned Chambolle-Pock solver for HK-dEIT.  (v7: eta_op nuisance)
%
%   [rho, m, eta, info] = HK_CORE(A, W, Y, G, o)
%
%   min_{rho>=0, m, eta}  sum_t 1/2 ||W^(1/2)( Y_t - A*(EL*rho_t + eta_t) )||^2
%                       + alpha * Action_delta(rho, m)
%                       + beta/2 * sum_t ||D rho_t||^2
%                       + gamma/2 * sum_t ||eta_t||^2
%
%   Action_delta = sum_k dt * [ sum_e dia_e * m_{k,e}^2 / (2*theta_{k,e})
%                 + delta^2 * sum_i vol_i * zeta_{k,i}^2 / (2*rb_{k,i}) ]
%   with  rb_k = (rho_k + rho_{k+1})/2,
%         theta_{k,e} = (rb_{k,i} + rb_{k,j})/2,
%         zeta_k = (rho_{k+1} - rho_k)/dt + divG * m_k   (NOT independent).
%
%   A : M x Ne   linearized forward map, SIGN SUCH THAT  Y ~ A*(EL*rho+eta)
%   W : M x M    measurement inverse covariance (SPD; speye(M) is fine)
%   Y : M x T+1  difference data, one column per frame
%   G : dual graph from calc_dual_graph
%   o : options
%        .alpha,.delta,.beta,.gamma,.dt   model weights (delta > 0, physical
%                                          length units)
%        .max_iter (3000)
%        .tol      (1e-4)   stop when max(rP, rD) < tol, where rP, rD are the
%                           RELATIVE preconditioned primal/dual residuals
%                           (step-size independent; see v4 notes)
%        .verbose  (200)    print every so many iterations, 0 = silent
%        .use_eta  (true)   false -> eta = 0 (N x N quadratic prox)
%        .eta_op   ([])     v7: M x K data-space operator of the nuisance eta,
%                           Y ~ A*EL*rho + eta_op*eta, penalty gamma/2 ||eta||^2,
%                           eta K x T+1 (no positivity, no transport). Use it for
%                           non-lung tissue clusters (A*Phi) and electrode-movement
%                           Jacobian columns. Empty -> eta_op = A (element eta).
%                           Scaled by 1/||A|| together with A when normalize.
%        .normalize(true)   scale A,Y by 1/||A||
%        .tau0     (1)      primal step; pick it with diag_cp_E4.m
%        .init     ('tikhonov') or 'zero'
%        .nondim   (true)   lengths in units of h = mean(G.d) internally
%        .scale_alpha (true) alpha_eff = alpha * rms(Y)   (P3 equivariance)
%        .solver   ('auto') quadratic prox: 'dense' | 'woodbury' | 'auto'
%                           (auto = woodbury if use_eta=false and N > 2M)
%        .adapt    (true)   v5: adaptive primal/dual balance every 50 it
%                           (Goldstein et al.): if rP > ratio*rD, tau *= 1/(1-a)
%                           and all sigma /= 1/(1-a); the reverse if rD >
%                           ratio*rP; a <- a*eta. tau*sigma is invariant, so the
%                           CP condition (T4) is unchanged. The balance needed
%                           varies by ~4 decades across the alpha grid of E4,
%                           which a single fixed tau0 cannot match.
%        .adapt_ratio (2) .adapt_a0 (0.5) .adapt_eta (0.95)
%        .vent_w   ([])     v6: N x 1 reference ventilation profile w >= 0.
%                           The Hellinger source becomes zeta - w*c(t) with a
%                           free c(t): tidal filling distributed like w is
%                           not penalized, only source HETEROGENEITY is.
%                           Outputs info.vent_c (1 x T, physical volume
%                           rate), info.zeta_res (penalized part).
%        .rho0     (0)      v6: gas content at the reference frame (scalar or
%                           N x 1, units of rho). Action and positivity act on
%                           rho0 + rho (absolute content) instead of rho.
%
%   v4 notes
%   * Progress of CP iterates must NOT be judged by the primal objective
%     (the perspective terms are huge/+Inf at slightly infeasible iterates)
%     nor by the relative change (it scales with tau). We use the residuals
%     of the optimality system  0 in dF(u) + K'q,  0 in dG*(q) - K u :
%        P = (u_k - u_{k+1})/tau                      in dF(u_{k+1}) + K'q_{k+1}
%        D = S^-1 (q_k - q_{k+1}) + K(ub_k - u_{k+1}) in dG*(q_{k+1}) - K u_{k+1}
%     rP = ||P|| / max(||K'q||, ||P - K'q||),  rD = ||D|| / max(||Ku||, ||D + Ku||).
%     rP >> rD : primal lags -> larger tau0;  rD >> rP : smaller tau0.
%   * Woodbury: I + tau*(A_L' W A_L + beta D'D) = Q + U (tau W) U' with
%     Q sparse and rank(U) <= M, so no dense N x N matrix is formed.
%
%   Outputs: rho (N x T+1), m (E x T, physical units), eta (Ne x T+1), info:
%     .zeta .res=[rP rD] .relchg .iters .hist=[it fit rP rD] .time=[K prox solve]
%     .tau .sig_e .sig_n .sig_p .L2scaled .Gn .h .alpha_eff .delta_nd .solver
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

% ---------------- defaults ----------------
if ~isfield(o,'max_iter');    o.max_iter = 3000;    end
if ~isfield(o,'tol');         o.tol = 1e-4;         end
if ~isfield(o,'verbose');     o.verbose = 200;      end
if ~isfield(o,'use_eta');     o.use_eta = true;     end
if ~isfield(o,'normalize');   o.normalize = true;   end
if ~isfield(o,'tau0');        o.tau0 = 1;           end
if ~isfield(o,'init');        o.init = 'tikhonov';  end
if ~isfield(o,'nondim');      o.nondim = true;      end
if ~isfield(o,'scale_alpha'); o.scale_alpha = true; end
if ~isfield(o,'solver');      o.solver = 'auto';    end
if ~isfield(o,'adapt');       o.adapt = true;       end   % v5 adaptive balance
if ~isfield(o,'adapt_ratio'); o.adapt_ratio = 2;    end
if ~isfield(o,'adapt_a0');    o.adapt_a0 = 0.5;     end
if ~isfield(o,'adapt_eta');   o.adapt_eta = 0.95;   end
if ~isfield(o,'vent_w');      o.vent_w = [];        end   % v6 ventilation mode
if ~isfield(o,'rho0');        o.rho0 = 0;           end   % v6 background gas
if ~isfield(o,'eta_op');      o.eta_op = [];        end   % v7 nuisance operator
if o.delta <= 0; error('hk_core: delta must be > 0'); end

M  = size(A, 1);
N  = G.N;  E = G.E;
T1 = size(Y, 2);  T = T1 - 1;
dt = o.dt;

if o.normalize
    s = normest(A, 1e-4);
    A = A / s;  Y = Y / s;
    if ~isempty(o.eta_op); o.eta_op = o.eta_op / s; end
end

% ---------------- non-dimensionalization ----------------
% (i) lengths in units of h = mean centroid distance (equivalent to a
%     rescaling of alpha, and of beta in 3-D; delta converted here);
% (ii) alpha_eff = alpha * rms(Y): Y -> cY gives (rho, m) -> c(rho, m).
h = 1;
dim = round(median(G.ell .* G.d ./ G.dia));          % 2 (2-D) or 3 (3-D)
if o.nondim
    h   = mean(G.d);
    G.ell  = G.ell / h^(dim-1);
    G.d    = G.d   / h;
    G.dia  = G.dia / h^dim;
    G.vol  = G.vol / h^dim;
    G.divG = spdiags(1./G.vol,0,N,N) * G.B' * spdiags(G.ell,0,E,E);
    G.D    = spdiags(sqrt(G.ell./G.d),0,E,E) * G.B;
end
G.absB = abs(G.B);                                   % cached for hk_K/hk_Kt

% ---- v6: ventilation mode and background gas content ----
% (a) vent_w (N x 1, >= 0): reference ventilation profile w. The source in
%     the Hellinger term becomes zeta - w*c(t) with c(t) FREE, so a source
%     distributed like w (homogeneous-like tidal filling) costs nothing and
%     only the HETEROGENEITY of the source is penalized. w is normalized to
%     vol'*w = 1 (non-dim volumes): c(t)*dt = volume explained by the mode.
%     zeta - w c is linear in (rho, m, c) -> the problem stays convex.
% (b) rho0 (scalar or N x 1, >= 0): gas content at the reference frame
%     (FRC), in the units of rho. The action and the positivity act on the
%     ABSOLUTE content rho0 + rho (rho = change w.r.t. the reference frame):
%     with rho0 = 0 no gas can be transported out of the reference frame
%     (theta = 0) and regions cannot lose gas below their reference level.
%     Implemented as an affine shift of K: prox of [G(.+b)]* at q is the
%     prox of G* at q + sigma*b, so only the dual updates change.
%     For the calibration equivariance (P3) rho0 must scale with the data.
use_vent = ~isempty(o.vent_w);
if use_vent
    wv = max(double(o.vent_w(:)), 0);
    if numel(wv) ~= N || ~any(wv); error('hk_core: vent_w must be N x 1, >= 0, not all 0'); end
    G.wv = wv / (G.vol' * wv);                       % vol'*w = 1 (physical meaning)
    % v6.1 conditioning: with vol'*w = 1 the entries of w are ~1/N, so the
    % c-column of K was tiny vs the rho-columns (~1/dt) and c barely moved
    % (residuals ~5e-3 after 4000 it). Reparametrize c_int = c/sc so that
    % ||w_int|| = sqrt(2)/dt (a rho-column's zeta entries). The minimizer
    % in (rho, m) is unchanged; outputs are converted back.
    sc_vent = (sqrt(2)/dt) / norm(G.wv);
    G.wv = G.wv * sc_vent;
else
    sc_vent = 1;
end
r0 = double(o.rho0(:));
if isscalar(r0); r0 = r0 * ones(N, 1); end
if numel(r0) ~= N || any(r0 < 0); error('hk_core: rho0 must be a scalar or N x 1, >= 0'); end
th0 = 0.5 * (G.absB * r0);                           % shift of the theta row
delta = o.delta / h;
alpha = o.alpha;
if o.scale_alpha
    alpha = alpha * sqrt(mean(Y(:).^2));
end

% ---------------- data-side quantities ----------------
AEL = A * G.EL;                  % M x N   (U' in the Woodbury notation)
if isempty(o.eta_op); AE = A; else; AE = o.eta_op; end   % data-space operator of eta
Keta = size(AE, 2);
WY  = W * Y;
Bl  = AEL' * WY;                 % N x T1
DtD = G.D' * G.D;                % N x N sparse

% ---------------- preconditioned step sizes ----------------
schur = @(X) sqrt(norm(X, 1) * norm(X, inf));    % ||X||_2 upper bound
L_th = 0.5 * schur(G.absB);
L_ze = 2/dt + schur(G.divG);
sg0_e = 1 / max(L_th, 1)^2;              % group (theta, m)
sg0_n = 1 / max(L_ze, 1)^2;              % group (rb, zeta)
sg0_p = 1;                               % positivity copy
tau   = o.tau0;

rs = RandStream('mt19937ar', 'Seed', 7);
pu.rho = randn(rs, N, T1);  pu.m = randn(rs, E, T);  pu.eta = zeros(Keta, T1);
pu.c   = randn(rs, 1, T) * use_vent;
lam = 0;
for it = 1:500
    pq = hk_K(pu, G, dt);
    pq.th = sg0_e*pq.th;  pq.m  = sg0_e*pq.m;
    pq.ze = sg0_n*pq.ze;  pq.rb = sg0_n*pq.rb;
    pq.pos = sg0_p*pq.pos;
    pw = hk_Kt(pq, G, dt);
    nn = sqrt(norm(pw.rho,'fro')^2 + norm(pw.m,'fro')^2 + norm(pw.c)^2);
    if nn == 0; break; end
    pu.rho = pw.rho/nn;  pu.m = pw.m/nn;  pu.c = pw.c/nn;
    lam_old = lam;  lam = nn;
    if abs(lam - lam_old) <= 1e-9*lam; break; end
end
f     = 0.98 / (tau * lam * 1.05);       % 5% margin (power iter. under-estimates)
sig_e = f * sg0_e;
sig_n = f * sg0_n;
sig_p = f * sg0_p;

% ---------------- quadratic prox: (I + tau*H) X = RHS ----------------
useW = ~o.use_eta && ( strcmpi(o.solver,'woodbury') || ...
                       (strcmpi(o.solver,'auto') && N > 2*M) );
% tau-independent pieces, so the factorization can be rebuilt cheaply when
% the adaptive rule changes tau (make_qsolver)
QS.N = N;  QS.Ne = Keta;  QS.beta = o.beta;  QS.gamma = o.gamma;  QS.DtD = DtD;
if o.use_eta
    QS.kind = 'dense-eta';
    WA     = W * AE;
    QS.Hll = AEL' * (W * AEL);
    QS.Hle = AEL' * WA;                  % N x K
    QS.Hee = AE'  * WA;                  % K x K (dense; K = Ne without eta_op)
    Be     = AE' * WY;                   % K x T1
elseif useW
    QS.kind = 'woodbury';
    QS.U    = AEL';                      % N x M
    QS.Winv = inv(full(W));
else
    QS.kind = 'dense';
    QS.Hll  = AEL' * (W * AEL);
end
solveQ = make_qsolver(QS, tau);
solver_name = QS.kind;

% ---------------- dual prox weights ----------------
ce = (alpha * dt / sig_e) * G.dia;                   % E x 1
ci = (alpha * delta^2 * dt / sig_n) * G.vol;         % N x 1

% ---------------- init ----------------
u.rho = zeros(N, T1);  u.m = zeros(E, T);  u.eta = zeros(Keta, T1);
u.c   = zeros(1, T);
if strcmpi(o.init, 'tikhonov')
    % clipped per-frame Tikhonov, lam0 = 1% of mean diag(H); Woodbury form
    U0   = AEL';                                     % N x M
    UtU  = U0' * U0;                                 % M x M
    trH  = sum(sum(UtU .* full(W)'));                % trace(U W U')
    lam0 = 1e-2 * trH / N;
    tmp  = (lam0 * inv(full(W)) + UtU) \ (U0' * Bl);
    u.rho = max((Bl - U0*tmp) / lam0, -r0);          % (lam0 I + U W U')\Bl, rho0+rho >= 0
end
ub = u;
q.th = zeros(E,T); q.m = zeros(E,T); q.ze = zeros(N,T);
q.rb = zeros(N,T); q.pos = zeros(N,T1);

hist = [];  relchg = NaN;  rP = NaN;  rD = NaN;
tim  = zeros(1,3);                                   % [K/K', prox, solve]
mon  = 50;
ad   = o.adapt_a0;  nadapt = 0;  tau_hist = tau;

% ---------------- preconditioned Chambolle-Pock loop ----------------
for it = 1:o.max_iter
    chk = (mod(it, mon) == 0) || (it == o.max_iter);
    if chk; q_old = q; ub_old = ub; end

    % dual ascent + prox of G* (Moreau), per-group steps
    t0 = tic;  k = hk_K(ub, G, dt);  tim(1) = tim(1) + toc(t0);
    q.th  = q.th  + sig_e*(k.th + th0);          % + affine shift (rho0)
    q.m   = q.m   + sig_e*k.m;
    q.ze  = q.ze  + sig_n*k.ze;
    q.rb  = q.rb  + sig_n*(k.rb + r0);
    q.pos = q.pos + sig_p*(k.pos + r0);

    t0 = tic;
    [thp, mp]  = prox_perspective(q.th/sig_e, q.m/sig_e,  ce);
    [rbp, zep] = prox_perspective(q.rb/sig_n, q.ze/sig_n, ci);
    tim(2) = tim(2) + toc(t0);
    q.th = q.th - sig_e*thp;   q.m  = q.m  - sig_e*mp;
    q.rb = q.rb - sig_n*rbp;   q.ze = q.ze - sig_n*zep;
    q.pos = min(q.pos, 0);               % Moreau of indicator{rho0+rho >= 0}

    % primal descent + prox of F
    u0 = u;
    t0 = tic;  kt = hk_Kt(q, G, dt);  tim(1) = tim(1) + toc(t0);
    vr = u.rho - tau*kt.rho;
    vm = u.m   - tau*kt.m;
    t0 = tic;
    if o.use_eta
        X  = solveQ([vr; u.eta] + tau*[Bl; Be]);     % kt.eta = 0
        u.rho = X(1:N, :);
        u.eta = X(N+1:end, :);
    else
        u.rho = solveQ(vr + tau*Bl);
    end
    tim(3) = tim(3) + toc(t0);
    u.m = vm;
    u.c = u.c - tau*kt.c;                % c is unpenalized: prox = identity

    % over-relaxation
    ub.rho = 2*u.rho - u0.rho;
    ub.m   = 2*u.m   - u0.m;
    ub.eta = 2*u.eta - u0.eta;
    ub.c   = 2*u.c   - u0.c;

    % ---------------- convergence monitor ----------------
    if chk
        % primal residual P = (u_k - u_{k+1})/tau
        P.rho = (u0.rho - u.rho)/tau;  P.m = (u0.m - u.m)/tau;  P.eta = (u0.eta - u.eta)/tau;
        P.c   = (u0.c - u.c)/tau;
        dF.rho = P.rho - kt.rho;  dF.m = P.m - kt.m;  dF.eta = P.eta;   % in dF(u_{k+1})
        dF.c   = P.c - kt.c;
        rP = nu(P) / max([nu(kt), nu(dF), eps]);

        % dual residual D = S^-1 (q_k - q_{k+1}) + K(ub_k - u_{k+1})
        du.rho = ub_old.rho - u.rho;  du.m = ub_old.m - u.m;  du.eta = ub_old.eta - u.eta;
        du.c   = ub_old.c - u.c;
        kd = hk_K(du, G, dt);
        ku = hk_K(u,  G, dt);
        ku.th = ku.th + th0;  ku.rb = ku.rb + r0;  ku.pos = ku.pos + r0;   % K u + b
        D.th  = (q_old.th  - q.th )/sig_e + kd.th;
        D.m   = (q_old.m   - q.m  )/sig_e + kd.m;
        D.ze  = (q_old.ze  - q.ze )/sig_n + kd.ze;
        D.rb  = (q_old.rb  - q.rb )/sig_n + kd.rb;
        D.pos = (q_old.pos - q.pos)/sig_p + kd.pos;
        dG.th = D.th + ku.th;  dG.m = D.m + ku.m;  dG.ze = D.ze + ku.ze;
        dG.rb = D.rb + ku.rb;  dG.pos = D.pos + ku.pos;               % in dG*(q_{k+1})
        rD = nq(D) / max([nq(ku), nq(dG), eps]);

        % adaptive primal/dual balance (Goldstein et al.): keep tau*sigma
        % fixed (CP condition unchanged), move tau towards rP ~ rD, with a
        % geometrically shrinking adaptation level ad (-> convergence kept)
        if o.adapt && isfinite(rP) && isfinite(rD) && rP > 0 && rD > 0
            fct = 0;
            if     rP > o.adapt_ratio * rD;  fct = +1;   % primal lags -> larger tau
            elseif rD > o.adapt_ratio * rP;  fct = -1;   % dual lags   -> smaller tau
            end
            if fct ~= 0
                sm    = (1 - ad)^(-fct);
                tau   = tau * sm;
                sig_e = sig_e / sm;  sig_n = sig_n / sm;  sig_p = sig_p / sm;
                ce    = (alpha * dt / sig_e) * G.dia;
                ci    = (alpha * delta^2 * dt / sig_n) * G.vol;
                solveQ = make_qsolver(QS, tau);
                ad     = ad * o.adapt_eta;
                nadapt = nadapt + 1;  tau_hist(end+1) = tau; %#ok<AGROW>
            end
        end

        relchg = sqrt( (nu(P)*tau)^2 / (nu(u)^2 + eps) );
        if o.verbose && (mod(it, o.verbose) == 0 || it == o.max_iter)
            R   = Y - AEL*u.rho - AE*u.eta;
            fit = 0.5 * sum(sum(R .* (W*R)));
            hist(end+1,:) = [it, fit, rP, rD]; %#ok<AGROW>
            fprintf('hk_core: it %5d  fit %.4e  rP %.2e  rD %.2e  min(rho0+rho) %.1e\n', ...
                    it, fit, rP, rD, min(min(u.rho + r0)));
        end
        if max(rP, rD) < o.tol; break; end
    end
end

rho = u.rho;  eta = u.eta;
info.zeta   = diff(rho, 1, 2)/dt + G.divG * u.m;    % TOTAL source (invariant under nondim)
if use_vent
    info.zeta_res  = info.zeta - G.wv * u.c;        % part penalized by Hellinger
    info.vent_c    = u.c * sc_vent * h^dim;         % mode amplitude: PHYSICAL volume rate
    info.vent_w    = G.wv / sc_vent / h^dim;        % profile per PHYSICAL volume (vol'*w = 1)
else
    info.zeta_res  = info.zeta;
    info.vent_c    = [];
    info.vent_w    = [];
end
info.rho0 = r0;
m = h * u.m;                                        % flux in PHYSICAL units
info.res    = [rP rD];
info.relchg = relchg;
info.iters  = it;
info.hist   = hist;
info.time   = tim;
info.tau    = tau;
info.sig_e  = sig_e;  info.sig_n = sig_n;  info.sig_p = sig_p;
info.L2scaled = tau * lam;
info.Gn = G;  info.h = h;  info.alpha_eff = alpha;  info.delta_nd = delta;
info.solver = solver_name;
info.nadapt = nadapt;  info.tau_hist = tau_hist;
end

% ======================================================================
function solveQ = make_qsolver(QS, tau)
% factorization of the quadratic prox  (I + tau*H) X = RHS  for a given tau
switch QS.kind
    case 'dense-eta'
        N = QS.N;
        Mat = [ eye(N) + tau*(QS.Hll + QS.beta*full(QS.DtD)),  tau*QS.Hle ; ...
                tau*QS.Hle',  eye(QS.Ne) + tau*(QS.Hee + QS.gamma*eye(QS.Ne)) ];
        solveQ = dense_solver((Mat + Mat')/2);
    case 'woodbury'
        Q = speye(QS.N) + (tau*QS.beta) * QS.DtD;    % sparse SPD
        [Rq, flag, Pq] = chol(Q, 'vector');          % Q(Pq,Pq) = Rq'*Rq
        if flag; error('hk_core: Q not SPD'); end
        Z  = qsolve(QS.U, Rq, Pq);                   % Q \ U
        Cm = QS.Winv/tau + QS.U'*Z;   Cm = (Cm + Cm')/2;
        Rc = chol(Cm);
        U  = QS.U;
        solveQ = @(X) woodbury_apply(X, Rq, Pq, Z, U, Rc);
    otherwise
        Mat = eye(QS.N) + tau*(QS.Hll + QS.beta*full(QS.DtD));
        solveQ = dense_solver((Mat + Mat')/2);
end
end

% ======================================================================
function s = dense_solver(Mat)
try
    Pf = decomposition(Mat, 'chol');     % MATLAB R2017b+
    s = @(X) Pf \ X;
catch
    Rc = chol(Mat);
    s = @(X) Rc \ (Rc' \ X);
end
end

function Xo = qsolve(X, Rq, Pq)
Xo = zeros(size(X));
Xo(Pq,:) = Rq \ (Rq' \ X(Pq,:));
end

function Xo = woodbury_apply(X, Rq, Pq, Z, U, Rc)
% (Q + U C U')^-1 X = Q^-1 X - Z (C^-1 + U' Z)^-1 U' Q^-1 X,  Z = Q^-1 U
Y0 = qsolve(X, Rq, Pq);
Xo = Y0 - Z * (Rc \ (Rc' \ (U' * Y0)));
end

function v = nu(s)
v2 = norm(s.rho,'fro')^2 + norm(s.m,'fro')^2 + norm(s.eta,'fro')^2;
if isfield(s, 'c'); v2 = v2 + norm(s.c)^2; end
v = sqrt(v2);
end

function v = nq(s)
v = sqrt(norm(s.th,'fro')^2 + norm(s.m,'fro')^2 + norm(s.ze,'fro')^2 + ...
         norm(s.rb,'fro')^2 + norm(s.pos,'fro')^2);
end
