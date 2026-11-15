function test_hk_unit
%TEST_HK_UNIT  Unit tests for the HK-dEIT building blocks (pure MATLAB).
%
%   Run:  test_hk_unit
%   Tests:
%     T1  prox_perspective vs brute-force minimization + edge cases
%     T2  calc_dual_graph identities: B*1 = 0, vol'*divG = 0, dia,ell > 0
%     T3  adjointness of hk_K / hk_Kt
%     T4  operator norm (power iteration inside hk_core) vs explicit SVD
%     T5  volume budget: vol'*(rho_{k+1}-rho_k) = dt * vol'*zeta_k
%     T6  hk_core smoke test: convex problem converges, rho >= -tol
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

fprintf('=== HK-dEIT unit tests ===\n');
rng(0);

% ------------------------------------------------- T1: prox_perspective
nfail = 0;
for trial = 1:300
    bt = 4*randn; at = 4*randn; c = 10.^(2*randn);
    [b, a] = prox_perspective(bt, at, c);
    % brute force over b >= 0 with inner closed-form a(b) = b*at/(b+c);
    % at b=0 the convention gives a=0 and perspective term 0.
    bg = linspace(0, abs(bt)+abs(at)+5*c+5, 200001);
    ag = bg.*at./(bg+c);
    fg = 0.5*(bg-bt).^2 + 0.5*(ag-at).^2 + c*ag.^2./(2*max(bg,realmin));
    fg(1) = 0.5*bt^2 + 0.5*at^2;
    [fmin, ix] = min(fg);
    if b > 0;  fprox = 0.5*(b-bt)^2 + 0.5*(a-at)^2 + c*a^2/(2*b);
    else;      fprox = 0.5*bt^2 + 0.5*at^2;
    end
    if fprox > fmin + 1e-6*max(1,abs(fmin)) + 1e-9
        nfail = nfail + 1;
        fprintf('  T1 FAIL: bt=%.3f at=%.3f c=%.3g  prox f=%.8g grid f=%.8g (b*=%.4g)\n',...
                bt, at, c, fprox, fmin, bg(ix));
    end
    % stationarity residual when b > 0
    if b > 1e-10
        res = (b - bt).*(b + c).^2 - (c/2)*at^2;
        assert(abs(res) < 1e-6*max(1,(b+c)^2), 'T1: cubic residual too large');
    end
end
% edge cases
[b,a] = prox_perspective(-3, 0, 1);   assert(b==0 && a==0);
[b,a] = prox_perspective( 2, 0, 1);   assert(abs(b-2)<1e-12 && a==0);
[b,a] = prox_perspective(-5, 1e-8, 1);assert(b>=0);
fprintf('T1 prox_perspective: %s (%d/300 brute-force fails)\n', pf(nfail==0), nfail);

% T1b: v2 (Cardano start) vs v1 (pure monotone Newton) over wide scales
% and near-degenerate cubics (bt ~ -c, at ~ 0, huge at)
n1b = 20000;
sc  = @(n) sign(randn(n,1)) .* 10.^(2*randn(n,1));
bt1 = sc(n1b);  at1 = sc(n1b);  c1 = 10.^(2*randn(n1b,1));
k3  = 1:2000;   bt1(k3) = -c1(k3) .* (1 + 1e-6*randn(2000,1));   % p ~ 0
k4  = 2001:4000; at1(k4) = 1e-8 * randn(2000,1);                  % r ~ 0
[b2, a2] = prox_perspective(bt1, at1, c1);
[b1, a1] = prox_v1(bt1, at1, c1);
s1b = 1e-10 * (abs(bt1) + abs(at1) + c1);            % per-case problem scale
e1b = max([abs(b2-b1) ./ max(abs(b1), s1b); abs(a2-a1) ./ max(abs(a1), s1b)]);
fprintf('T1b prox v2 vs v1 (20000 wide-scale cases): %s (max rel dev %.1e)\n', ...
        pf(e1b < 1e-7), e1b);

% ------------------------------------------------- small structured mesh
fmdl = mk_grid_mesh(6, 5);          % 6x5 cells -> 60 triangles
G = calc_dual_graph(fmdl, []);
% ------------------------------------------------- T2: graph identities
e1 = norm(G.B*ones(G.N,1));
e2 = norm(ones(1,G.N)*spdiags(G.vol,0,G.N,G.N)*G.divG);   % vol'*divG = 0
ok2 = e1 < 1e-12 && e2 < 1e-10 && all(G.dia>0) && all(G.ell>0) && all(G.vol>0);
fprintf('T2 dual graph: %s (||B*1||=%.1e, ||vol''*divG||=%.1e, N=%d E=%d)\n',...
        pf(ok2), e1, e2, G.N, G.E);

% ------------------------------------------------- T3: adjointness
T1n = 7; dt = 0.11;
u.rho = randn(G.N, T1n); u.m = randn(G.E, T1n-1); u.eta = randn(G.Ne, T1n);
q.th = randn(G.E,T1n-1); q.m = randn(G.E,T1n-1); q.ze = randn(G.N,T1n-1);
q.rb = randn(G.N,T1n-1); q.pos = randn(G.N,T1n);
Ku  = hk_K(u, G, dt);
Ktq = hk_Kt(q, G, dt);
ip1 = inq(Ku, q);  ip2 = inu(Ktq, u);
ok3 = abs(ip1-ip2) < 1e-9*max(1, abs(ip1));
fprintf('T3 adjoint <Ku,q> vs <u,K''q>: %s (%.6e vs %.6e)\n', pf(ok3), ip1, ip2);

% T3b: adjoint with the v6 ventilation mode (extra primal block c, 1 x T)
Gw = G;  Gw.wv = rand(G.N, 1);
uw = u;  uw.c = randn(1, T1n-1);
Kuw  = hk_K(uw, Gw, dt);
Ktqw = hk_Kt(q, Gw, dt);
ip1w = inq(Kuw, q);  ip2w = inu(Ktqw, uw) + Ktqw.c * uw.c';
ok3b = abs(ip1w-ip2w) < 1e-9*max(1, abs(ip1w));
fprintf('T3b adjoint with ventilation mode: %s (%.6e vs %.6e)\n', pf(ok3b), ip1w, ip2w);

% ------------------------------------------------- T4: CP step condition
% The quantity that matters for convergence of the preconditioned CP in
% hk_core is  tau * ||S^(1/2) K||_2^2 < 1  with the ACTUAL steps used.
% Check it exactly with an explicit K, for several tau0.
nE = G.E*(T1n-1);  nN = G.N*(T1n-1);
M4 = 20;
A4 = randn(M4, G.Ne)/sqrt(M4);
Y4 = A4 * (G.EL * abs(randn(G.N, T1n)));
o4 = struct('alpha',1e-2,'delta',3,'beta',1e-3,'gamma',10,'dt',dt, ...
            'max_iter',1,'tol',0,'verbose',0,'use_eta',false,'normalize',true);
ok4 = true;  str4 = '';
for tau0 = [0.1 1 10]
    o4.tau0 = tau0;
    [~, ~, ~, i4] = hk_core(A4, speye(M4), Y4, G, o4);
    Kmat = explicit_K(i4.Gn, dt, T1n);    % K on the graph hk_core really uses
    sw =[i4.sig_e*ones(2*nE,1); i4.sig_n*ones(2*nN,1); i4.sig_p*ones(G.N*T1n,1)];
    cnd = i4.tau * norm(sqrt(sw) .* Kmat, 2)^2;
    ok4 = ok4 && (cnd < 1);
    str4 = [str4 sprintf(' tau0=%g: %.4f', tau0, cnd)]; %#ok<AGROW>
end
fprintf('T4 CP condition tau*||S^1/2 K||^2 < 1: %s (%s)\n', pf(ok4), str4);

% ------------------------------------------------- T5: volume budget
ze = diff(u.rho,1,2)/dt + G.divG*u.m;
lhs = G.vol' * diff(u.rho,1,2);
rhs = dt * (G.vol' * ze);
ok5 = norm(lhs - rhs) < 1e-10 * max(1, norm(lhs));
fprintf('T5 volume budget: %s (err %.2e)\n', pf(ok5), norm(lhs-rhs));

% ------------------------------------------------- T6: hk_core smoke test
M = 30;
A = randn(M, G.Ne)/sqrt(M);
rho_t = zeros(G.N, 5);
rho_t(round(G.N/2)+(-2:2), :) = 1;          % static blob
Yc = A * (G.EL * rho_t);
o = struct('alpha',1e-2,'delta',3,'beta',1e-3,'gamma',10,'dt',0.1, ...
           'max_iter',20000,'tol',1e-5,'verbose',0,'use_eta',false, ...
           'normalize',true);
[rho_h, m_h, ~, infoc] = hk_core(A, speye(M), Yc, G, o);
rel = norm(rho_h - rho_t,'fro')/norm(rho_t,'fro');
ok6 = (min(rho_h(:)) > -1e-6) && (rel < 0.5) && (max(infoc.res) < 1e-5);
fprintf(['T6 hk_core smoke: %s (rel err %.3f, min rho %.1e, %d iters, ', ...
         'rP %.1e rD %.1e, ||m|| %.2e - static target => small flux)\n'], ...
        pf(ok6), rel, min(rho_h(:)), infoc.iters, infoc.res(1), infoc.res(2), ...
        norm(m_h,'fro'));

% ------------------------------------------------- T7: calibration (P3)
% With scale_alpha = true, Y -> c*Y must give (rho, m) -> c*(rho, m)
% EXACTLY (every CP iterate is equivariant, so no convergence needed).
o7 = o;  o7.max_iter = 300;  o7.tol = 0;
[r1, m1] = hk_core(A, speye(M), Yc,     G, o7);
[r2, m2] = hk_core(A, speye(M), 7.3*Yc, G, o7);
e7 = sqrt(norm(r2-7.3*r1,'fro')^2 + norm(m2-7.3*m1,'fro')^2) / ...
     sqrt(norm(7.3*r1,'fro')^2 + norm(7.3*m1,'fro')^2);
ok7 = e7 < 1e-8;
fprintf('T7 calibration equivariance Y->cY => rho->c*rho: %s (rel dev %.1e)\n', pf(ok7), e7);

% ------------------------------------------------- T8: Woodbury == dense
% the two quadratic-prox solvers must give the same CP iterates
o8 = o;  o8.max_iter = 200;  o8.tol = 0;
o8.solver = 'dense';     [r_d, m_d] = hk_core(A, speye(M), Yc, G, o8);
o8.solver = 'woodbury';  [r_w, m_w, ~, i8] = hk_core(A, speye(M), Yc, G, o8);
e8 = sqrt(norm(r_d-r_w,'fro')^2 + norm(m_d-m_w,'fro')^2) / ...
     sqrt(norm(r_d,'fro')^2 + norm(m_d,'fro')^2);
ok8 = e8 < 1e-7 && strcmp(i8.solver, 'woodbury');
fprintf('T8 Woodbury vs dense quadratic prox: %s (rel dev %.1e)\n', pf(ok8), e8);

% ------------------------------------------------- T9: voxel-cluster graph
fm9 = mk_grid_mesh(12, 10);                          % 240 triangles
grp9 = 1 + (mean(reshape(fm9.nodes(fm9.elems',1),3,[]))' > 6);  % 2 groups
G9 = calc_dual_graph(fm9, [], struct('target_N', 30, 'groups', grp9));
Gf9 = calc_dual_graph(fm9, []);
c9a = norm(G9.B*ones(G9.N,1));
c9b = norm(ones(1,G9.N)*spdiags(G9.vol,0,G9.N,G9.N)*G9.divG);
c9c = abs(sum(G9.vol) - sum(Gf9.vol)) / sum(Gf9.vol);
c9d = all(full(sum(G9.EL,2)) == 1);                  % each element in 1 cluster
mix = accumarray(G9.cluster, grp9(G9.node_idx), [], @(x) numel(unique(x)));
c9e = all(mix == 1);                                 % no cluster mixes groups
ok9 = c9a < 1e-12 && c9b < 1e-10 && c9c < 1e-12 && c9d && c9e && ...
      abs(G9.N - 30) <= 0.25*30;
% ------------------------------------------------- T10: rho0 shift (v6)
% with beta = 0, solving with background rho0 and data Y must equal solving
% with rho0 = 0 and data Y + A*rho0, up to rho' = rho + rho0 (same problem
% after substitution). scale_alpha off (it would see different rms(Y)).
% A CONSTANT rho0 satisfies D*rho0 = 0 (B*1 = 0), so the equivalence also
% holds with beta > 0; rho0 = 1 keeps rho0 + rho away from 0 (well posed).
r0c = 1.0;
o10 = struct('alpha',1e-2,'delta',3,'beta',1e-2,'gamma',10,'dt',0.1, ...
             'max_iter',40000,'tol',1e-7,'verbose',0,'use_eta',false, ...
             'normalize',true,'scale_alpha',false);
rt10 = rho_t;  rt10(1:5, 3:5) = rt10(1:5, 3:5) - 0.3;        % dips below reference
Y10  = A * (G.EL * rt10);
o10.rho0 = r0c;  [ra, ~, ~, ia] = hk_core(A, speye(M), Y10, G, o10);
o10.rho0 = 0;    [rb, ~, ~, ib] = hk_core(A, speye(M), Y10 + A*(G.EL*(r0c*ones(G.N, size(Y10,2)))), G, o10);
e10 = norm(ra + r0c - rb, 'fro') / norm(rb, 'fro');
conv10 = max([ia.res ib.res]) < 1e-5;
ok10 = e10 < 1e-3 && min(ra(:)) >= -r0c - 1e-6;
if ok10; st10 = 'PASS'; elseif ~conv10; st10 = 'INCONCLUSIVE (not converged)'; else; st10 = 'FAIL'; end
fprintf('T10 rho0 shift equivalence: %s (rel dev %.1e, min rho %.3f >= -rho0 = %.2f, res a %.0e b %.0e)\n', ...
        st10, e10, min(ra(:)), -r0c, max(ia.res), max(ib.res));

fprintf('T9 voxel clusters: %s (N=%d of target 30, E=%d, ||B*1||=%.0e, ||vol''*divG||=%.0e, vol err %.0e, groups kept %d)\n', ...
        pf(ok9), G9.N, G9.E, c9a, c9b, c9c, c9e);

% ------------------------------------------------- T11: eta_op (v7)
% (a) eta_op = A must reproduce the default element-eta iterates exactly
o11 = o;  o11.use_eta = true;  o11.max_iter = 200;  o11.tol = 0;
[r11a, m11a] = hk_core(A, speye(M), Yc, G, o11);
o11.eta_op = A;  [r11b, m11b] = hk_core(A, speye(M), Yc, G, o11);
e11 = sqrt(norm(r11a-r11b,'fro')^2 + norm(m11a-m11b,'fro')^2) / ...
      sqrt(norm(r11a,'fro')^2 + norm(m11a,'fro')^2);
% (b) a data-space nuisance B*xi added to the data is absorbed by eta_op = B
rs11 = RandStream('mt19937ar', 'Seed', 11);
B11  = randn(rs11, M, 3) / sqrt(M);  xi = randn(rs11, 3, size(Yc,2)) * norm(Yc,'fro')/sqrt(numel(Yc));
o11b = o;  o11b.use_eta = true;  o11b.eta_op = B11;  o11b.gamma = 1e-8;
[r11c, ~, eta11, i11] = hk_core(A, speye(M), Yc + B11*xi, G, o11b);
[r11d] = hk_core(A, speye(M), Yc + B11*xi, G, o);                      % no nuisance
rel11c = norm(r11c - rho_t,'fro')/norm(rho_t,'fro');
rel11d = norm(r11d - rho_t,'fro')/norm(rho_t,'fro');
ok11 = e11 < 1e-10 && rel11c < 0.5 && rel11c < rel11d && size(eta11,1) == 3;
fprintf(['T11 eta_op: %s (eta_op=A vs default rel dev %.1e | nuisance data: rel err with eta_op %.3f, ' ...
         'without %.3f, %d iters res %.0e)\n'], pf(ok11), e11, rel11c, rel11d, i11.iters, max(i11.res));

fprintf('=== done ===\n');
end

% ======================================================================
function [b, a] = prox_v1(bt, at, c)
% reference: v1 prox (monotone Newton from a proven upper start)
c = c + zeros(size(bt));
at2 = at.^2;  btp = max(bt, 0);
b = btp + (c/2).*at2 ./ (btp + c).^2;
for it = 1:500
    bc = b + c;
    g  = (b - bt).*bc.^2 - (c/2).*at2;
    gp = bc.^2 + 2*(b - bt).*bc;
    st = g ./ max(gp, realmin);
    b  = b - st;
    if all(abs(st(:)) <= 1e-15 * (abs(b(:)) + c(:))); break; end
end
neg = (b <= 0) | ~isfinite(b);
b(neg) = 0;
a = b .* at ./ (b + c);
a(neg) = 0;
end

function s = pf(ok); if ok; s='PASS'; else; s='FAIL'; end; end

function ip = inq(q1, q2)
ip = q1.th(:)'*q2.th(:) + q1.m(:)'*q2.m(:) + q1.ze(:)'*q2.ze(:) + ...
     q1.rb(:)'*q2.rb(:) + q1.pos(:)'*q2.pos(:);
end

function ip = inu(u1, u2)
ip = u1.rho(:)'*u2.rho(:) + u1.m(:)'*u2.m(:) + u1.eta(:)'*u2.eta(:);
end

function fmdl = mk_grid_mesh(nx, ny)
% structured triangular mesh on [0,nx] x [0,ny] (pure MATLAB)
[X, Y] = meshgrid(0:nx, 0:ny);
nodes = [X(:) Y(:)];
id = @(i,j) (i-1)*(ny+1) + j;      % column-major of meshgrid output
elems = zeros(2*nx*ny, 3); k = 0;
for i = 1:nx
    for j = 1:ny
        a = id(i,j); b = id(i+1,j); c = id(i,j+1); d = id(i+1,j+1);
        k = k+1; elems(k,:) = [a b c];
        k = k+1; elems(k,:) = [b d c];
    end
end
fmdl.type  = 'fwd_model';
fmdl.nodes = nodes;
fmdl.elems = elems;
end

function Kmat = explicit_K(G, dt, T1n)
% build K as an explicit matrix by applying hk_K to unit vectors
nu = G.N*T1n + G.E*(T1n-1) + G.Ne*T1n;
nq = G.E*(T1n-1)*2 + G.N*(T1n-1)*2 + G.N*T1n;
Kmat = zeros(nq, nu);
for jj = 1:nu
    z = zeros(nu,1); z(jj) = 1;
    u = vec2u(z, G, T1n);
    q = hk_K(u, G, dt);
    Kmat(:,jj) = [q.th(:); q.m(:); q.ze(:); q.rb(:); q.pos(:)];
end
end

function u = vec2u(z, G, T1n)
n1 = G.N*T1n;  n2 = G.E*(T1n-1);
u.rho = reshape(z(1:n1), G.N, T1n);
u.m   = reshape(z(n1+1:n1+n2), G.E, T1n-1);
u.eta = reshape(z(n1+n2+1:end), G.Ne, T1n);
end
