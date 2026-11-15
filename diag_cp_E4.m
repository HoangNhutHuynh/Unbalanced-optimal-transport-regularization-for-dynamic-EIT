%DIAG_CP_E4  Pick the Chambolle-Pock step balance tau0 on the E4 problem. (v2)
%
%   The CP condition holds for ANY tau0 (test T4), but the speed depends on
%   the primal/dual balance. Criterion (hk_core v4): the relative
%   preconditioned primal/dual residuals rP, rD after a fixed number of
%   iterations - lower max(rP,rD) = closer to the optimum. NOT the primal
%   objective (meaningless at infeasible CP iterates) and NOT relchg
%   (scales with tau).   rP >> rD -> increase tau0;  rD >> rP -> decrease.
%
%   Needs EIDORS on the path. Builds E4_setup.mat via run_E4_sim (setup
%   only) if it is not found on the MATLAB path / current folder.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

if ~exist('E4_setup.mat', 'file') || isempty(whos('-file', 'E4_setup.mat', 'Gc'))
    E4_SETUP_ONLY = true;  %#ok<NASGU>     % (re)build: needs the cluster graph Gc
    run_E4_sim;
end
clear E4_SETUP_ONLY
load E4_setup.mat

img_bkg = calc_jacobian_bkgnd(imdl);
A = -calc_jacobian(img_bkg);
W = calc_meas_icov(imdl);
Y = calc_difference_data(VH, VI, imdl.fwd_model);
rel_lung = @(DS) norm(DS(lung_idx,:) - ds_true(lung_idx,:), 'fro') ...
               / norm(ds_true(lung_idx,:), 'fro');
fprintf(['problem size: M = %d meas, Ne = %d elems, T+1 = %d\n' ...
         '  fine graph N = %d, E = %d  ->  cluster graph N = %d, E = %d (used here)\n'], ...
        size(A,1), size(A,2), size(Y,2), G.N, G.E, Gc.N, Gc.E);

o = struct('alpha',1e-5,'delta',0.15*diam,'beta',1e-3,'gamma',1e3, ...
           'dt',dt,'max_iter',5000,'tol',0,'verbose',1000, ...
           'use_eta',false,'normalize',true, ...
           'adapt',false);      % diagnostic of FIXED steps (hk_core v5 adapts by default)

tau_list = [0.3 3 30 300];
tab = zeros(numel(tau_list), 6);
for i = 1:numel(tau_list)
    o.tau0 = tau_list(i);
    fprintf('\n--- tau0 = %g ---\n', o.tau0);
    t_all = tic;
    [rho, ~, ~, info] = hk_core(A, W, Y, Gc, o);
    tsec = toc(t_all);
    tab(i,:) = [o.tau0, info.res(1), info.res(2), rel_lung(-(Gc.EL*rho)), ...
                1e3*tsec/info.iters, info.iters];
    fprintf('    solver %s | %.1f ms/it  [K/K'' %.0f%%, prox %.0f%%, solve %.0f%%]\n', ...
            info.solver, 1e3*tsec/info.iters, 100*info.time/tsec);
end

fprintf('\n============ CP step balance (%d it, same problem) ============\n', o.max_iter);
fprintf('%8s %10s %10s %10s %10s\n', 'tau0', 'rP', 'rD', 'rel err', 'ms/it');
for i = 1:size(tab,1)
    fprintf('%8g %10.2e %10.2e %10.4f %10.1f\n', tab(i,1:5));
end
[~, ib] = min(max(tab(:,2), tab(:,3)));
fprintf('\n-> best tau0 = %g (lowest max(rP,rD)).\n', tab(ib,1));
if ib == 1 || ib == numel(tau_list)
    fprintf('   It is at an END of tau_list: extend the list and re-run before using it.\n');
end
