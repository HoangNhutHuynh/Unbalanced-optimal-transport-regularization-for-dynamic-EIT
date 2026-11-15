%TOY_HK_1D  1-D mechanism test of HK-dEIT vs L2 space-time and Tikhonov. (v3)
%
%   Pure MATLAB, no EIDORS. M = 12 measurements, N = 40 cells, T+1 = 16
%   frames. One block translating 1 cell/frame (constant mass 4) and one
%   block growing from 0 to mass 4 (pure source). Baselines get an ORACLE
%   parameter sweep (best rel. error vs ground truth).
%
%   v3 changes (after v2 run):
%     - two noise levels (1%, 5%). At 1% with M < N the oracle drifts to the
%       near-exact-fit limit (lower-edge alpha / lambda_t): there only the
%       RATIOS of the priors matter. 5% gives an interior optimum.
%     - hk_core v3 non-dimensionalizes and uses alpha_eff = alpha*rms(Y),
%       so alpha grids are comparable with run_E4_sim.
%     - mechanism metrics for claim C1 (moving block, cells 1..27):
%         ghost  = fraction of positive mass farther than 2 cells from the
%                  true block, averaged over frames (0 = no ghosting)
%         COM    = RMS error of the block centre of mass [cells]
%       and for the growing block: final mass (true 4) and HK source.
%
%   Run:   toy_hk_1d          (about 2x the v2 run time)
%   Saves: toy_results.mat
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clear; clc;

noise_list = [0.01 0.05];

% ---------------- problem setup (fixed across noise levels) -------------
rng(1);
N  = 40;  M = 12;  T1 = 16;  dt = 0.1;
A  = randn(M, N) / sqrt(M);
Wm = speye(M);

E = N - 1;
G.N = N; G.E = E; G.Ne = N;
G.B    = spdiags([ones(E,1) -ones(E,1)], [0 1], E, N);
G.ell  = ones(E,1);  G.d = ones(E,1);  G.dia = 0.5*ones(E,1);
G.vol  = ones(N,1);
G.divG = spdiags(1./G.vol,0,N,N) * G.B' * spdiags(G.ell,0,E,E);
G.D    = spdiags(sqrt(G.ell./G.d),0,E,E) * G.B;
G.EL   = speye(N);
G.edges = [(1:E)' (2:E+1)'];

rho_true = zeros(N, T1);
move_cells = [];  pos_k = zeros(1,T1);
for k = 1:T1
    pos_k(k) = 5 + (k-1);
    rho_true(pos_k(k):pos_k(k)+3, k) = 1;
    move_cells = union(move_cells, pos_k(k):pos_k(k)+3);
end
grow_cells = 30:33;
for k = 1:T1
    rho_true(grow_cells, k) = (k-1)/(T1-1);
end
Yc = A * rho_true;
noise0 = randn(M, T1);                      % same noise pattern, scaled

relerr = @(X) norm(X - rho_true, 'fro') / norm(rho_true, 'fro');
mech   = @(X) mech_metrics(X, pos_k, grow_cells);

% grids
alphas = logspace(-8, 0, 9);
deltas = [0.5 1 2 5];
betas  = [0 logspace(-5, 0, 6)];
lts = logspace(-9, 3, 13);   lss = logspace(-9, 2, 12);
las = logspace(-8, 1, 19);

base_o = struct('gamma',1e6,'dt',dt,'tol',1e-4,'verbose',0, ...
                'use_eta',false,'normalize',true);
As  = sparse(A);  AtA = As'*As;
E1  = spdiags([-ones(T1,1) ones(T1,1)], [0 1], T1-1, T1);
Lt  = E1'*E1;     DtD = G.D'*G.D;

allres = struct();
for in = 1:numel(noise_list)
    nl = noise_list(in);
    Y  = Yc + nl * (norm(Yc,'fro')/sqrt(numel(Yc))) * noise0;
    Aty = As'*Y;
    fprintf('\n######## noise = %.0f%% ########\n', 100*nl);

    % ---- HK coarse sweep ----
    oc = base_o;  oc.max_iter = 4000;
    gl = [];  nrun = 0;  ntot = numel(alphas)*numel(deltas)*numel(betas);
    fprintf('HK coarse sweep: %d runs x 4000 it\n', ntot);
    for al = alphas
      for de = deltas
        for be = betas
            o = oc;  o.alpha = al;  o.delta = de;  o.beta = be;
            rho_h = hk_core(A, Wm, Y, G, o);
            gl(end+1,:) = [al de be relerr(rho_h)]; %#ok<SAGROW>
            nrun = nrun + 1;
            if mod(nrun, 50) == 0; fprintf('  run %3d/%3d\n', nrun, ntot); end
        end
      end
    end
    orf = base_o;  orf.max_iter = 40000;  orf.tol = 1e-6;
    b.hk  = refine_top(gl(gl(:,3)>0 ,:), 3, A, Wm, Y, G, orf, relerr);
    b.hk0 = refine_top(gl(gl(:,3)==0,:), 2, A, Wm, Y, G, orf, relerr);
    warn_edge('HK alpha', b.hk.alpha, alphas, true);
    warn_edge('HK delta', b.hk.delta, deltas, false);
    warn_edge('HK beta',  b.hk.beta,  betas(2:end), true);

    % ---- L2 space-time oracle ----
    b.l2 = struct('err', Inf);
    for lt = lts
      for ls = lss
          x  = (kron(speye(T1), AtA + ls*DtD) + lt*kron(Lt, speye(N))) \ Aty(:);
          er = relerr(reshape(x, N, T1));
          if er < b.l2.err
              b.l2 = struct('err',er,'lt',lt,'ls',ls,'rho',reshape(x,N,T1));
          end
      end
    end
    warn_edge('L2-ST lambda_t', b.l2.lt, lts, true);
    warn_edge('L2-ST lambda_s', b.l2.ls, lss, true);

    % ---- Tikhonov oracle ----
    b.tik = struct('err', Inf);
    for la = las
        Xt = (AtA + la*speye(N)) \ Aty;
        er = relerr(Xt);
        if er < b.tik.err; b.tik = struct('err',er,'lambda',la,'rho',Xt); end
    end
    warn_edge('Tikhonov lambda', b.tik.lambda, las, false);

    % ---- HK source attribution ----
    zeta    = diff(b.hk.rho,1,2)/dt + G.divG*b.hk.m;
    src_int = dt * sum(zeta, 2);
    s_move  = G.vol(move_cells)' * src_int(move_cells);
    s_grow  = G.vol(grow_cells)' * src_int(grow_cells);

    % ---- report ----
    meths = {'HK',               b.hk.rho,  b.hk.err;
             'HK beta=0',        b.hk0.rho, b.hk0.err;
             'L2 space-time',    b.l2.rho,  b.l2.err;
             'Tikhonov',         b.tik.rho, b.tik.err};
    fprintf('\n=========== TOY 1-D RESULTS (v3), noise %.0f%% ===========\n', 100*nl);
    fprintf('%-16s %8s %8s %10s %12s\n', 'Method', 'relerr', 'ghost', 'COM[cell]', 'grow mass');
    for r = 1:size(meths,1)
        mm = mech(meths{r,2});
        fprintf('%-16s %8.3f %8.3f %10.3f %12.3f\n', meths{r,1}, meths{r,3}, ...
                mm.ghost, mm.com, mm.grow_end);
    end
    fprintf('(truth: ghost 0, COM 0, grow mass 4.000)\n');
    fprintf('HK  : alpha=%.0e delta=%g beta=%.0e | %d it, relchg %.1e, alpha_eff %.2e\n', ...
            b.hk.alpha, b.hk.delta, b.hk.beta, b.hk.info.iters, b.hk.info.relchg, ...
            b.hk.info.alpha_eff);
    fprintf('L2  : lt=%.0e ls=%.0e (ratio lt/ls = %.1e)\n', b.l2.lt, b.l2.ls, b.l2.lt/b.l2.ls);
    fprintf('Tik : lambda=%.0e\n', b.tik.lambda);
    fprintf('HK source: moving region %+.4f (true 0) | growing region %+.4f (true 4)\n', ...
            s_move, s_grow);
    fprintf('min(rho_HK) = %.2e\n', min(b.hk.rho(:)));

    b.grid = gl;  b.s_move = s_move;  b.s_grow = s_grow;  b.noise = nl;
    allres(in).b = b; %#ok<SAGROW>

    figure('Name', sprintf('toy_hk_1d noise %.0f%%', 100*nl));
    subplot(2,2,1); imagesc(rho_true');  title('ground truth'); xlabel('cell'); ylabel('frame');
    subplot(2,2,2); imagesc(b.hk.rho');  title(sprintf('HK (err %.3f)',   b.hk.err));
    subplot(2,2,3); imagesc(b.l2.rho');  title(sprintf('L2-ST (err %.3f)',b.l2.err));
    subplot(2,2,4); imagesc(b.tik.rho'); title(sprintf('Tikhonov (err %.3f)',b.tik.err));
    colormap(parula);
end

save toy_results.mat allres rho_true G A Yc noise_list dt pos_k grow_cells

% ======================================================================
function mm = mech_metrics(X, pos_k, grow_cells)
% ghosting / centre-of-mass error for the moving block (cells 1..27) and
% final mass of the growing block
T1 = size(X,2);  cells = (1:27)';
Xp = max(X(cells,:), 0);
gh = zeros(1,T1);  ce = zeros(1,T1);
for k = 1:T1
    w   = (cells >= pos_k(k)-2) & (cells <= pos_k(k)+5);   % block +/- 2
    tot = sum(Xp(:,k)) + eps;
    gh(k) = sum(Xp(~w,k)) / tot;
    ce(k) = (cells' * Xp(:,k)) / tot - (pos_k(k) + 1.5);
end
mm.ghost    = mean(gh);
mm.com      = sqrt(mean(ce.^2));
mm.grow_end = sum(X(grow_cells, end));
end

function b = refine_top(gl, ntop, A, Wm, Y, G, orf, relerr)
[~, ord] = sort(gl(:,4));
b = struct('err', Inf);
for r = 1:min(ntop, size(gl,1))
    o = orf;
    o.alpha = gl(ord(r),1);  o.delta = gl(ord(r),2);  o.beta = gl(ord(r),3);
    fprintf('refine: alpha=%.0e delta=%g beta=%.0e ...', o.alpha, o.delta, o.beta);
    [rho_h, m_h, ~, infoh] = hk_core(A, Wm, Y, G, o);
    er = relerr(rho_h);
    fprintf(' err %.4f (%d it, relchg %.1e)\n', er, infoh.iters, infoh.relchg);
    if er < b.err
        b = struct('err',er,'alpha',o.alpha,'delta',o.delta,'beta',o.beta, ...
                   'rho',rho_h,'m',m_h,'info',infoh);
    end
end
end

function warn_edge(name, val, grid, lower_is_exact_fit)
lo = val <= min(grid)*(1+1e-12);  hi = val >= max(grid)*(1-1e-12);
if lo && lower_is_exact_fit
    fprintf('  note: %s = %.0e at LOWER edge -> near exact-fit limit (only prior ratios matter; expected at low noise)\n', name, val);
elseif lo || hi
    warning('%s = %.2e on the edge of its grid [%.0e, %.0e] - extend the grid.', ...
            name, val, min(grid), max(grid));
end
end
