function img = inv_solve_diff_hk(imdl, vh, vi)
%INV_SOLVE_DIFF_HK  Hellinger-Kantorovich (unbalanced OT) dynamic EIT solver.
%
%   img = inv_solve( imdl, vh, vi )   with  imdl.solve = @inv_solve_diff_hk
%
%   vh : homogeneous (reference) data - EIDORS data struct or M x 1 vector
%   vi : inhomogeneous data           - struct array or M x (T+1) matrix
%
%   Required options in  imdl.inv_solve_diff_hk :
%     .alpha, .delta, .beta, .gamma, .dt   (see hk_core.m)
%     .lung      element indices of the lung ROI on imdl.fwd_model, e.g.
%                [fmdl.mat_idx{2}; fmdl.mat_idx{3}], or a functional ROI
%                (TIV >= 20% of max, TREND-style). [] -> all elements.
%     .max_iter, .tol, .verbose, .use_eta, .normalize, .tau0   (optional)
%     .G          pre-built dual graph (skips calc_dual_graph), or
%     .graph_opts options for calc_dual_graph, e.g.
%                 struct('target_N',800,'groups',grp) for voxel clusters
%
%   Output image:
%     img.elem_data = -(EL*rho + eta)    (Ne x T+1, conductivity change)
%     img.hk.rho    (N  x T+1) air content on the lung dual graph
%                   (graph nodes = elements, or voxel clusters)
%     img.hk.flux   (E  x T)   edge fluxes m
%     img.hk.source (N  x T)   zeta from the continuity equation
%     img.hk.eta    (Ne x T+1) non-transport residual
%     img.hk.G      dual graph struct;  img.hk.info  solver info
%
%   NOTES
%   * The Jacobian is computed on imdl.fwd_model (no coarse2fine mapping in
%     this version); the dual graph lives on the same elements.
%   * We deliberately do NOT use inv_solve_core / the multi-frame branch of
%     GN_update: it builds kron(eye(nf), J'*W*J) ("massive memory hog") and
%     only supports the quadratic LLt time prior.
%   * eta is defined on ALL elements (size Ne), so the per-frame quadratic
%     solve is (N+Ne) x (N+Ne); set use_eta=false for an N x N solve.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

o = imdl.inv_solve_diff_hk;
if ~isfield(o, 'lung'); o.lung = []; end

% linearized forward map; sign so that  Y ~ A*(EL*rho + eta), rho ~ -dsigma
img_bkg = calc_jacobian_bkgnd(imdl);
J = calc_jacobian(img_bkg);
A = -J;

W = calc_meas_icov(imdl);
Y = calc_difference_data(vh, vi, imdl.fwd_model);   % M x (T+1)

if isfield(o, 'G') && ~isempty(o.G)
    G = o.G;                                   % pre-built (e.g. clustered)
else
    if ~isfield(o, 'graph_opts'); o.graph_opts = struct(); end
    G = calc_dual_graph(imdl.fwd_model, o.lung, o.graph_opts);
end
o = rmfield_if(o, 'G');                        % keep the option struct light

[rho, m, eta, info] = hk_core(A, W, Y, G, o);

img = eidors_obj('image', 'solved by inv_solve_diff_hk');
img.fwd_model = imdl.fwd_model;
img.elem_data = -(G.EL*rho + eta);
img.hk.rho    = rho;
img.hk.flux   = m;
img.hk.source = info.zeta;
img.hk.eta    = eta;
img.hk.G      = G;
img.hk.info   = info;
end

function s = rmfield_if(s, f)
if isfield(s, f); s = rmfield(s, f); end
end
